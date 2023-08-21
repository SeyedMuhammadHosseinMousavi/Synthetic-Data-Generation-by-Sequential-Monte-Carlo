%% Synthetic Data Generation by Sequential Monte Carlo (SMC)
% SMC methods, also known as particle filters, are used to estimate state variables in
% dynamic systems. They can also be adapted for generating synthetic data by iteratively 
% sampling from a sequence of distributions.
clear;
% Load the original dataset
load fisheriris
NF= size(meas); NF=NF(1,2); % Number of features 
Classes=3; % Number of classes
for i=1:NF
original_data=meas(:,i);
Target(1:50)=1;Target(51:100)=2;Target(101:150)=3;Target=Target'; % Original labels
num_original_data = numel(original_data);

% Parameters for the Sequential Monte Carlo simulation
num_particles = 10000; % Number of particles
num_samples = 300;   % Number of synthetic data points to generate

% Initialize particles with original data points
particles = datasample(original_data, num_particles, 'Replace', true);
% Initialize weights
weights = ones(1, num_particles) / num_particles;
% Preallocate array for synthetic data
synthetic_data = zeros(1, num_samples);
% Sequential Monte Carlo algorithm
for t = 1:num_samples
% Resample particles based on weights
resampled_particles = datasample(particles, num_particles, 'Weights', weights);
% Generate synthetic data point based on resampled particles
synthetic_data(t) = resampled_particles(randi(num_particles));
% Update weights based on distance between particles and synthetic data
distances = abs(synthetic_data(t) - resampled_particles);
weights = exp(-distances);
weights = weights / sum(weights);
end
Syn(:,i)=synthetic_data;
end

%% Getting labels of synthetic generated data by K-means clustering
[Lbl,C,sumd,D] = kmeans(Syn,Classes,'MaxIter',1000,...
'Display','final','Replicates',5);
% Generated data plus labels 
AugAll=[Syn Lbl];

%% Plot data and classes
Feature1=1;
Feature2=2;
f1=meas(:,Feature1); % feature1
f2=meas(:,Feature2); % feature 2
ff1=Syn(:,Feature1); % feature1
ff2=Syn(:,Feature2); % feature 2
figure('units','normalized','outerposition',[0 0 1 1])
subplot(2,2,1)
plot(meas, 'linewidth',1); title('Original Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,2)
plot(Syn, 'linewidth',1); title('Synthetic Data');
ax = gca; ax.FontSize = 12; ax.FontWeight='bold'; grid on;
subplot(2,2,[3 4])
histogram(original_data, 'Normalization', 'probability', 'DisplayName', 'Original Data');
hold on;
histogram(synthetic_data, 'Normalization', 'probability', 'DisplayName', 'Synthetic Data');
legend();
xlabel('Value');
ylabel('Probability');
title('Original vs. Synthetic Data Distribution');
%% Train and Test
% Training augmented dataset by SVM
% Training multiple times for getting average of them
TrainNumbers=2; % Number of trains
for i = 1:TrainNumbers
Mdlsvm = fitcecoc(Syn,Lbl);
CVMdlsvm = crossval(Mdlsvm);
SVMError(i) = kfoldLoss(CVMdlsvm);
SVMAccAugTrainAvg(i) = (1 - SVMError(i))*100;
disp ([' Training SVM No "',num2str(i)]);
end
SVMAccAugTrain=sum(SVMAccAugTrainAvg)/TrainNumbers; % Train accuracy
% Predict new data by augmented model (SVM) on the original dataset
[label5,score5,cost5] = predict(Mdlsvm,meas);
% Test error and accuracy calculations
DataSize=size(meas);DataSize=DataSize(1,1);
a=0;b=0;c=0;
for i=1:DataSize
if label5(i)== 1
a=a+1;
elseif label5(i)==2
b=b+1;
else
label5(i)==3
c=c+1;
end;end;
erra=abs(a-50);errb=abs(b-50);errc=abs(c-50);
err=erra+errb+errc;TestErr=err*100/DataSize;SVMAccAugTest=100-TestErr; % Test Accuracy
% Train and Test Accuracy Results
AugRessvm = [' Augmented Train SVM "',num2str(SVMAccAugTrain),'" Augmented Test SVM"', num2str(SVMAccAugTest),'"'];
disp(AugRessvm);
