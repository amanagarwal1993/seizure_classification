subj = 1;
ictal_len = 218;
nonictal_len = 600;
test_len = 931;
fs = 5000; 
alldata = [];
testdata = [];

%ictal
for i = 1:ictal_len
    fname = sprintf('patient_%d/ictal/patient_%d_%d.mat', subj,subj,i);
    matData = load(fname);
    data = matData.data;
    [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt] = featureExtract(data,fs);
    train_features = [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt,1];
    alldata = [alldata' train_features']';
end

%nonictal
for i = 1:nonictal_len
    fname = sprintf('patient_%d/non-ictal/patient_%d_%d.mat', subj,subj,i);
    matData = load(fname);
    data = matData.data;
    [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt] = featureExtract(data,fs);
    train_features = [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt,0];
    alldata = [alldata' train_features']';
end

%test
for i = 1:test_len
    fname = sprintf('patient_%d/test/patient_%d_test_%d.mat', subj,subj,i);
    matData = load(fname);
    data = matData.data;
    [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt] = featureExtract(data,fs);
    test_features = [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt];
    testdata = [testdata' test_features']';
end

csvwrite('train_featuresP1.csv', alldata)
csvwrite('test_featuresP1.csv', testdata)

%%%
function [E,L_line,Var,Activity, Complexity, Mobility, BetaPower, Kurt] = featureExtract(data,fs)
    [sample,channels] = size(data);

    E = zeros(ceil(sample/fs), channels); %Energy
    avg = zeros(ceil(sample/fs), channels); %The mean value for each window
    start_ind = zeros(ceil(sample/fs), 1); %The starting indice for each window
    len_ind = zeros(ceil(sample/fs), 1); %The duration of each window (except for last,
                                         % each window is 5000 data points)
    L_line = zeros(ceil(sample/fs), channels); %Line-length
    Var = zeros(ceil(sample/fs), channels); %Variance
    
    Activity = zeros(ceil(sample/fs), channels); %Variance
    Complexity = zeros(ceil(sample/fs), channels); %Variance
    Mobility = zeros(ceil(sample/fs), channels); %Variance

    BetaPower = zeros(ceil(sample/fs), channels);%Beta Power
    
    Kurt = zeros(ceil(sample/fs), channels);%Kurtosis
    
    j = 1; %counter variable for implementing new values into signal matrices

    for k = 1:fs:sample
        start = k; %start indice for window
        stop = min(k+fs-1,sample); %end indice for window
        len = stop-start+1; %length of window
    %     window range: [k, k+fs-1] 

          %The features are extracted through the given formulas (in case of
          %energy) or through matlab's own function (bandpower). The mean and
          %window indices are also determined for calculating line-length and
          %variance. For efficient computation, those values for calculated
          %in seperate loops (seen below)
        E(j, :) = (sum(data(start:stop,:).^2,1));
        avg(j, :) = (sum(data(start:stop,:),1))/len;
          
        xV = data(start:stop,:);
        xV(find(isnan(xV))) = []; %get rid of nan values
        
        %calculate Hjorth Parameters
        dxV = diff(xV);
        ddxV = diff(dxV);
        mx2 = mean(xV.^2);
        mdx2 = mean(dxV.^2);
        mddx2 = mean(ddxV.^2);
        mob = mdx2 / mx2;
        
        Activity(j, :) = mx2;
        Complexity(j, :) = sqrt(mddx2 / mdx2 - mob);
        Mobility(j, :) = sqrt(mob);
        
        %calculate beta power
        BetaPower(j,:) = bandpower(xV, fs, [12,30]);
        
        %calculate kurtosis
        Kurt(j,:) = kurtosis(xV);
        
        start_ind(j) = start;
        len_ind(j) = len;

        j = j+1;
    end

    for i = 1:length(start_ind) %This loops through the different windows
        start = start_ind(i); 
        stop = start + len_ind(i)-1;

        %Calculating Line-Length
        xi_0 = start_ind(i); %In formula, X[i-1]
        line_sum = 0;
        for w = start+1:stop %This loops through the different data points in each window
            line_sum = line_sum + abs(data(w,:)-data(xi_0,:));
            xi_0 = w; %updates X[i-1]
        end
        L_line(i, :) = line_sum;

        %Calculating Variance
        varsum = 0;
        for q = start:stop %This loops through the different data points in each window
            varsum = varsum + (data(q,:)-avg(i)).^2;
        end
        Var(i,:) = sqrt(varsum/len_ind(i));
    end

end


