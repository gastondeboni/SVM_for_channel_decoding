clear all
% options =  optimset('Display','off');
options = optimoptions('quadprog', ...
    'TolFun', 1e-12, ...        % Set the function tolerance for high precision
    'TolCon', 1e-12, ...        % Set the constraint tolerance for higher accuracy
    'TolX', 1e-12, ...          % Set the step size tolerance
    'MaxIter', 1000, ...        % Increase the maximum number of iterations
    'Display', 'off', ... % 'iter-detailed', ... % Display iteration details for monitoring
    'Algorithm', 'interior-point-convex'); % Use interior-point for better handling of constraints

%%  Parameters

% Choosing the type of modulation and SVM
modulation_type = '16QAM'; % 'BPSK', 'QPSK', '8PSK', '16QAM'
EbN0dB_gamma = -2:1:12; % gamma will be adapted for this Eb/N0
C = inf;
snr = [];

%% Do everything for each code
for CODE = ["Polar", "BCH"]

    % Load code
    
    % modulation order
    m = strcmp(modulation_type,'BPSK')*1 + strcmp(modulation_type,'QPSK')*2 +strcmp(modulation_type,'8PSK')*3 + strcmp(modulation_type,'16QAM')*4;
    
    % Generator matrix
    switch CODE
        case 'Polar'
            G = readmatrix('G_POLAR_32_11.txt');
        case 'BCH'
            G = readmatrix('G_BCH_32_11.txt');
    end
    
    k = size(G,1); n = size(G,2); 
    
    % Alphabet messages <=> Code words
    alpha_messages = de2bi(0:2^k-1, 'left-msb');
    ytrain = alpha_messages;
    alpha_codewords = mod(alpha_messages*G,2);
    
    % Assign these values to training samples
    s_train = modulate_codewords(alpha_codewords,modulation_type);
    
    if ~strcmp(modulation_type,'BPSK')
        xtrain = [real(s_train) imag(s_train)];
    else
        xtrain = s_train;
    end
    
    % The kernel has to be RBF to study gamma
    ker = @(x,y,gamma) exp(-gamma* pdist2(x,y,'euclidean').^2 );
    gamma = (10.^(EbN0dB_gamma/10) * (k/n) * m );
    
    %% Compute all of the SVMs for each gamma
    
    nb_train_samples = size(xtrain,1);
    
    alphas = zeros(nb_train_samples,k,length(gamma));
    ind = zeros(nb_train_samples,k);
    betas = zeros(k,length(gamma));
    Kx = zeros(2^k,2^k,length(gamma));
    for i = 1:length(gamma)
        Kx(:,:,i) = ker(xtrain,xtrain,gamma(i)); % Kernel matrix
    end
    
    % Solve optimization problems
    
    for i_gamma = 1:length(gamma)
        for i = 1:k
            %Define problem parameters for this region
            yt = ((ytrain(:,i) == 1)*1)*2-1;
            H = (yt*yt').*Kx(:,:,i_gamma);
            f = -ones(nb_train_samples,1);
            lb = zeros(nb_train_samples,1);
            ub = C*ones(nb_train_samples,1);
        
            % Solving the dual problem
            alpha = quadprog(H,f,[],[],yt',0,lb,ub,[],options);
            
            % Thresholding alphas and saving indexes
            ind(:,i) = alpha>1e-5; % removes alphas that are "numerically" equal to zero
            alpha(~ind(:,i)) = 0;
            alphas(:,i,i_gamma) = alpha;
            
            % Find the bias beta(i) 
            aux = logical(ind(:,i));
            [~,ii] = max(alphas(:,i,i_gamma));
            betas(i,i_gamma) = yt(ii) - sum( alphas(aux,i,i_gamma) .*yt(aux).* Kx(aux,ii,i_gamma) );
    
        end
        fprintf(['SVM ' num2str(i_gamma) ' done\n' ])
    end
    
    fprintf(['# of support vectors on each SVM: ' num2str(sum(ind)) '\n'])
    fprintf(['min and max alphas: ' num2str(min(min(alphas))) ' ' num2str(max(max(alphas))) '\n'])
    fprintf(['min and max beta: ' num2str(min(betas)) ' ' num2str(max(betas)) '\n'])
    
    %% SVM TESTING
    fprintf('Test... \n')
    
    % Parameters
    nb_words_test =  50000;
    error_margin = 5e-5;
    ber_obj = 1e-3;
    start_snr = 5;
    lr = 250;
    
    snr_gamma = zeros(length(gamma),1);
    for i_gamma = 1:length(gamma)
    
        fprintf('gamma of %d dB \n',EbN0dB_gamma(i_gamma))
        EbN0dB = start_snr;  
        ber = 1;
        while (abs(ber-ber_obj) > error_margin)
            
            sigma_2 = 1/(10^(EbN0dB/10) * (k/n) * m );
    
            % Random bit generation
            b = randi([0 1],k*nb_words_test,1);
    
            % Coding
            messages = reshape(b,k,[]).';
            codewords = mod(messages*G,2);
    
            % modulation and noise
            stest = modulate_codewords(codewords,modulation_type);
            w = sqrt(sigma_2/2)*(randn(size(stest)) + 1j*randn(size(stest)));
            s_noisy = stest + w;
    
            if ~strcmp(modulation_type,'BSPK')
                xtest = [real(s_noisy) imag(s_noisy)];
            else
                xtest = s_noisy;
            end
    
            % decode using SVM
            P = zeros(size(messages,1), size(messages,2));
            for i = 1:k
                yt = ((ytrain(:,i) == 1)*1)*2-1;
                ii = find(ind(:,i));
                P(:,i) = ((sum(alphas(ii,i,i_gamma) .* yt(ii) .* ker(xtrain(ii,:),xtest,gamma(i_gamma)) ).' +betas(i,i_gamma))>0)*1;
            end
            ber = sum(sum(P ~= messages)) / (k*nb_words_test);
    
            % Update EbN0dB
            % EbN0dB = EbN0dB + (ber-ber_obj)*lr;
            if abs(ber-ber_obj) > error_margin
                EbN0dB = EbN0dB + (ber-ber_obj)*lr;
            end
            fprintf(['EbN0 = ' num2str(EbN0dB) ' dB, BER = ' num2str(ber) ', error = ' num2str(ber-ber_obj) '\n'])
        end
        snr_gamma(i_gamma) = EbN0dB;
    
    end
    snr = [snr, snr_gamma];
end

figure
plot(EbN0dB_gamma, snr(:,1), Marker='o', DisplayName=sprintf("Polar"))
grid on, hold on
plot(EbN0dB_gamma, snr(:,2), Marker='o', DisplayName=sprintf("BCH"))
legend(Interpreter="latex")
xlabel('s (dB)')
ylabel('Eb/N0 (dB)')
title('Different SVMs')

% save BER
file = fopen('Studies/gamma_study.txt', 'w+');
for i = 1:2
    for j = 1:length(gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), snr(j,i));
    end
    fprintf(file, "\n");
end
