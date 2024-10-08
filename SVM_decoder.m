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
kernel_type = 'RBF'; % 'linear', 'poly', 'RBF', 'tanh', 'wave'
modulation_type = '16QAM'; % 'BPSK', 'QPSK', '8PSK', '16QAM'
d = 5; % Degree of the polynomial kernel if using poly kernel
EbN0dB_gamma = [0,5,10]; % gamma will be adapted for this Eb/N0
C = inf;
CODE = 'BCH'; % 'Polar' for Polar(32,11) and 'BCH' for BCH(32,11)

% Parameters for BER testing
EbN0dB_test = 0:1:8;
nb_words_test =  20000;
min_errors = 3000;
max_sims = 300000;

%% 

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
gamma = (10.^(EbN0dB_gamma/10) * (k/n) * m ); %exponential coefficient (large gamma => exponential drops rapidly)

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

%% Define kernel function according to choice
switch kernel_type
    case 'linear'
        ker = @(x,y,gamma) (x*y');
    case 'poly'
        ker = @(x,y,gamma) ((1 + x*y.').^d); %Polynomial kernel
    case 'RBF'
        ker = @(x,y,gamma) exp(-gamma* pdist2(x,y,'euclidean').^2 );
    case 'tanh'
        ker = @(x,y,gamma) tanh(x*y');
    case 'wave'
        ker = @(x,y,gamma)  sin(pdist2(x,y,'euclidean')) ./ (pdist2(x,y,'euclidean') + 0.001);
end

rbf = @(x,y,gamma) exp(-gamma* pdist2(x,y,'euclidean').^2 );

%% Training for each region
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
end

fprintf(['# of support vectors on each SVM: ' num2str(sum(ind)) '\n'])
fprintf(['min and max alphas: ' num2str(min(min(alphas))) ' ' num2str(max(max(alphas))) '\n'])
fprintf(['min and max beta: ' num2str(min(betas)) ' ' num2str(max(betas)) '\n'])

%% SVM TESTING

fprintf('Test every SVM... \n')

alphas_optimal = ones(size(alphas));
beta_optimal = zeros(size(betas));

ber_SVM = zeros(length(EbN0dB_test),length(gamma));
ber_SVM_perfect = zeros(1,length(EbN0dB_test));

for i_EbN0dB = 1:length(EbN0dB_test)
    fprintf('Eb/N0 = %d \n',EbN0dB_test(i_EbN0dB))

    EbN0dB = EbN0dB_test(i_EbN0dB);
    sigma_2 = 1/(10^(EbN0dB/10) * (k/n) * m );
    gamma_perfect = 1/(sigma_2);

    nb_errors = zeros(length(gamma),1); nb_errors_perfect = 0;
    nb_sims = 0;
    while (min(nb_errors) < min_errors) && (nb_sims < max_sims)

        % Random bit generation
        b = randi([0 1],k*nb_words_test,1);

        % Coding to polar
        messages = reshape(b,k,[]).';
        codewords = mod(messages*G,2);

        % modulation and noise
        stest = modulate_codewords(codewords,modulation_type);
        w = sqrt(sigma_2/2)*(randn(size(stest)) + 1j*randn(size(stest)));
        s_noisy = stest + w;

        if ~strcmp(modulation_type,'BPSK')
            xtest = [real(s_noisy) imag(s_noisy)];
        else
            xtest = real(s_noisy);
        end

        % Test the SVMs for each value of gamma
        for i_gamma = 1:length(gamma)
            P = zeros(size(messages,1), size(messages,2));
            for i = 1:k
                yt = ((ytrain(:,i) == 1)*1)*2-1;
                ii = find(ind(:,i));
                P(:,i) = ((sum(alphas(ii,i,i_gamma) .* yt(ii) .* ker(xtrain(ii,:),xtest,gamma(i_gamma)) ).' +betas(i,i_gamma))>0)*1;
            end
            nb_errors(i_gamma) = nb_errors(i_gamma) + sum(sum(P ~= messages));
        end

        % Perfect SVM = MAP
        P = zeros(size(messages,1), size(messages,2));
        for i = 1:k
            yt = ((ytrain(:,i) == 1)*1)*2-1;
            ii = find(ind(:,i));
            % P(:,i) = ((sum(alphas_optimal(ii,i) .* yt(ii) .* ker(xtrain(ii,:),xtest,gamma_perfect) ).' +beta_optimal(i))>0)*1;
            P(:,i) = ((sum(alphas_optimal(:,i) .* yt .* rbf(xtrain,xtest,gamma_perfect) ).')>0)*1;
        end
        nb_errors_perfect = nb_errors_perfect + sum(sum(P ~= messages));

        nb_sims = nb_sims + 1;

    end

    ber_SVM(i_EbN0dB,:) = nb_errors/(k*nb_words_test*nb_sims);
    ber_SVM_perfect(i_EbN0dB) = nb_errors_perfect/(k*nb_words_test*nb_sims);
end

% save('32-11/ber_SVM_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM')
% save('32-11/ber_SVM_gamma_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM_gamma')
% save('BCH-32-11/32-11/ber_SVM_alphas_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM_alphas')
% save('BCH-32-11/32-11/ber_SVM_perfect_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM_perfect')

% save('32-11/ber_SVM_POLAR_32_11.mat', 'EbN0dB_test', 'ber_SVM')
% save('32-11/ber_SVM_gamma_POLAR_32_11.mat', 'EbN0dB_test', 'ber_SVM_gamma')
% save('POLAR-32-11/ber_SVM_alphas_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM_alphas')
% save('POLAR-32-11/ber_SVM_perfect_BCH_32_11.mat', 'EbN0dB_test', 'ber_SVM_perfect')

% load 'ber_SVM_BCH_32_11.mat'
% load 'ber_SVM_gamma_BCH_32_11.mat'
% load 'ber_SVM_alphas_BCH_32_11.mat'
% load 'ber_SVM_perfect_BCH_32_11.mat'

figure
for i =1:length(gamma)
    semilogy(EbN0dB_test, ber_SVM(:,i), Marker='o', DisplayName=sprintf("$\\gamma$ adapted for $E_b/N_0=%d$dB", EbN0dB_gamma(i)))
    hold on, grid on
end
semilogy(EbN0dB_test, ber_SVM_perfect, Marker='o', DisplayName='MAP')
legend(Interpreter="latex")
xlabel('Eb/N0')
ylabel('BER')
title('Different SVMs')

% save BER
file = fopen(['BER_' CODE '.txt'], 'w+');
for i = 1:length(gamma)
    for j = 1:length(EbN0dB_test)
        fprintf(file, "%d   %f\\\\\n", EbN0dB_test(j), ber_SVM(j,i));
    end
    fprintf(file, "\n");
end
for j = 1:length(EbN0dB_test)
    fprintf(file, "%d   %f\\\\\n", EbN0dB_test(j), ber_SVM_perfect(j));
end

