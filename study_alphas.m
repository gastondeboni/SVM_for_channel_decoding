clear all
options =  optimset('Display','off');

%%  Parameters
% Choosing the type of modulation and SVM
modulation_type = '16QAM'; % 'BPSK', 'QPSK', '8PSK', '16QAM'
EbN0dB_gamma = -2:1:12; % gamma will be adapted for this Eb/N0
C = inf;
CODE_list = ["Polar", "BCH"];

%% Do everything for each code
i_code = 0;
n = 32; k = 11;
alphas = zeros(2^k,k,length(EbN0dB_gamma),2);
betas = zeros(k,length(EbN0dB_gamma),2);
for CODE = CODE_list

    i_code = i_code+1;
    
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
    ind = zeros(nb_train_samples,k);
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
            alphas(:,i,i_gamma,i_code) = alpha;
            
            % Find the bias beta(i) 
            aux = logical(ind(:,i));
            [~,ii] = max(alphas(:,i,i_gamma));
            betas(i,i_gamma,i_code) = yt(ii) - sum( alphas(aux,i,i_gamma) .*yt(aux).* Kx(aux,ii,i_gamma) );
    
        end
        fprintf(['SVM ' num2str(i_gamma) ' done\n' ])
    end
    
    fprintf(['# of support vectors on each SVM: ' num2str(sum(ind)) '\n'])
    fprintf(['min and max alphas: ' num2str(min(min(alphas))) ' ' num2str(max(max(alphas))) '\n'])
    fprintf(['min and max beta: ' num2str(min(betas)) ' ' num2str(max(betas)) '\n'])
    
end
%% Save everything

mean_alpha =  squeeze(mean(mean(alphas)));
min_alpha = squeeze(min(min(alphas)));
max_alpha = squeeze(max(max(alphas)));
mean_beta = squeeze(mean(betas));
min_beta = squeeze(min(betas));
max_beta = squeeze(max(betas));

figure
grid on, hold on
plot(EbN0dB_gamma, max_alpha(:,1), 'r', DisplayName=sprintf("Polar"))
plot(EbN0dB_gamma, mean_alpha(:,1), 'r')
plot(EbN0dB_gamma, min_alpha(:,1), 'r')
plot(EbN0dB_gamma, max_beta(:,1), 'r')
plot(EbN0dB_gamma, mean_beta(:,1), 'r')
plot(EbN0dB_gamma, min_beta(:,1), 'r')

plot(EbN0dB_gamma, max_beta(:,2), 'b', DisplayName=sprintf("BCH"))
plot(EbN0dB_gamma, mean_beta(:,2), 'b')
plot(EbN0dB_gamma, min_beta(:,2), 'b')
plot(EbN0dB_gamma, max_alpha(:,2), 'b')
plot(EbN0dB_gamma, mean_alpha(:,2), 'b')
plot(EbN0dB_gamma, min_alpha(:,2), 'b')

legend(Interpreter="latex")
xlabel('s (dB)')
ylabel('Eb/N0 (dB)')
title('Different SVMs')

% save
file = fopen('Studies/alpha_study.txt', 'w+');
for i = 1:2
    
    fprintf(file, "CODE: %s\n", CODE_list(i));

    fprintf(file, "Mean alpha\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), mean_alpha(j,i));
    end
    fprintf(file, "\n");

    fprintf(file, "Min alpha\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), min_alpha(j,i));
    end
    fprintf(file, "\n");

    fprintf(file, "Max alpha\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), max_alpha(j,i));
    end
    fprintf(file, "\n");

    fprintf(file, "Mean beta\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), mean_beta(j,i));
    end
    fprintf(file, "\n");

    fprintf(file, "Min beta\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), min_beta(j,i));
    end
    fprintf(file, "\n");

    fprintf(file, "Max beta\n");
    for j = 1:length(EbN0dB_gamma)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_gamma(j), max_beta(j,i));
    end
    fprintf(file, "\n");

end
