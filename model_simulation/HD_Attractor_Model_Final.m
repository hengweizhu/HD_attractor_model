function HD_Attractor_Model_Final(K_Inh_gain_list, K_Exc_gain_list, gaus_ratio_list, Visual_std_list, opts)
% HD_Attractor_Model_Final
% HD ring attractor simulation (Stringer et al. 2002-style).
%
% Required args:
%   K_Inh_gain_list, K_Exc_gain_list, gaus_ratio_list, Visual_std_list
%
% Optional opts (struct):
%   opts.shuffleDuringStationary : true/false (default false)
%   opts.shuffleFraction         : fraction of Wrot elements to shuffle [0..1] (default 1.0)
%   opts.shuffleMode             : 'normal' or 'head_restrained' (default 'normal')
%       - 'normal'          : no shuffling during stationary periods
%       - 'head_restrained' : shuffle Wrot during stationary periods
%   opts.seed                    : [] or integer seed (default [])
%   opts.saveDir                 : directory to save .mat files
%                                 (default "C:\home\...\HD_attractor_model\Simulations\")
%   opts.fileSuffix              : suffix like "_test.mat" (default "_test.mat")
%   opts.plotEvery               : plot every N steps, 0 disables live plot (default 10)
%   opts.duration                : total simulation time in seconds (default 600)
%   opts.dt                      : timestep in seconds (default 0.1)
%   opts.numSegmentsToChange     : number of stationary segments (default 20)
%   opts.segmentLength           : length of each stationary segment (in samples) (default 50)
%
% Notes:
%   - Create a Simulations folder to save outputs into and change the
%   opts.SaveDir to your own path.
%
% Outputs (saved):
%   hHD, rHD, decodedHD, Az, stat_startIndices, stat_endIndices, Wrot, new_Wrot_matrix,
%   K_Inh_gain, K_Exc_gain, gaus_ratio, Visual_std, opts

    if nargin < 5 || isempty(opts)
        opts = struct();
    end

    % ---- defaults ----
    opts = setDefault(opts, 'shuffleDuringStationary', false);
    opts = setDefault(opts, 'shuffleFraction', 1.0);
    opts = setDefault(opts, 'shuffleMode', 'normal'); % 'normal' or 'head_restrained'
    opts = setDefault(opts, 'seed', []);
    opts = setDefault(opts, 'saveDir', "C:\home\...\HD_attractor_model\Simulations\");
    opts = setDefault(opts, 'fileSuffix', "_test.mat");
    opts = setDefault(opts, 'plotEvery', 10);
    opts = setDefault(opts, 'duration', 600);
    opts = setDefault(opts, 'dt', 0.1);
    opts = setDefault(opts, 'numSegmentsToChange', 20);
    opts = setDefault(opts, 'segmentLength', 50);

    % ---- RNG seed ----
    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    % ---- constants ----
    N_HD_Neurons = 100;
    HD_Neurons_PD = (1:N_HD_Neurons) * 2*pi/N_HD_Neurons - pi;

    N_Velocity_Neurons = 101;
    Velocity_Neurons_Offset = (-50:50) * 0.02;

    K_Recurrent = 52 / N_HD_Neurons;
    K_Visual = 17;

    tau = 1;
    alpha = 30;
    beta  = 0.022;

    Recurrent_std = 0.15;
    Visual_normal_std = 0.262;

    % Calibration mapping for velocity neuron index
    Vneurons_Calibration = (-50:50) * 7.55;

    % Time
    dt = opts.dt;
    time = 0:dt:opts.duration;
    T = numel(time);

    % Low-pass filter for velocity
    [b_filt, a_filt] = butter(2, 2*dt/2, 'low');

    % Visual template
    Iv_norm = normpdf(HD_Neurons_PD, 0, Visual_normal_std)';

    % Simulation params
    RECCURENT_GAIN = 1;
    VELOCITY_GAIN = 1;

    for K_Inh_gain = K_Inh_gain_list
        for K_Exc_gain = K_Exc_gain_list
            for gaus_ratio = gaus_ratio_list % used for filename purpose only
                for Visual_std = Visual_std_list % used for filename purpose only

                    disp(K_Exc_gain);
                    disp(K_Inh_gain);

                    K_Inhibitory_base = K_Inh_gain * K_Recurrent;
                    K_Excitatory_base = K_Exc_gain * K_Recurrent;

                    % ---- Build structured Wrot once ----
                    Wrot = zeros(N_HD_Neurons, N_HD_Neurons, N_Velocity_Neurons);
                    for j = 1:N_HD_Neurons
                        for k = 1:N_Velocity_Neurons
                            delta_pd = HD_Neurons_PD - (HD_Neurons_PD(j) + Velocity_Neurons_Offset(k));
                            delta_pd = mod(delta_pd + pi, 2*pi) - pi;
                            Wrot(:, j, k) = normpdf(delta_pd', 0, Recurrent_std);
                        end
                    end
                    Wrot = Wrot / mvnpdf(0, 0, Recurrent_std); % peak=1

                    % ---- Random head velocity trajectory ----
                    Head_Velocity = randn(size(time)) * 240;
                    Vmax = max(Vneurons_Calibration);

                    Head_Velocity = filter(b_filt, a_filt, Head_Velocity);
                    Head_Velocity(Head_Velocity >  Vmax) =  Vmax;
                    Head_Velocity(Head_Velocity < -Vmax) = -Vmax;

                    arrayLength = numel(Head_Velocity);
                    segmentLength = opts.segmentLength;
                    numSegmentsToChange = opts.numSegmentsToChange;

                    indicesToChange = randperm(arrayLength - segmentLength + 1, numSegmentsToChange);

                    stat_startIndices = zeros(1, numSegmentsToChange);
                    stat_endIndices   = zeros(1, numSegmentsToChange);

                    for iSeg = 1:numSegmentsToChange
                        startIndex = indicesToChange(iSeg);
                        endIndex   = startIndex + segmentLength - 1;
                        Head_Velocity(startIndex:endIndex) = 0;
                        stat_startIndices(iSeg) = startIndex;
                        stat_endIndices(iSeg)   = endIndex;
                    end

                    % Integrate velocity -> azimuth (degrees)
                    Az = cumsum(Head_Velocity * dt);

                    % Velocity neuron index
                    Velocity_Index = interp1(Vneurons_Calibration, 1:101, VELOCITY_GAIN * Head_Velocity, 'nearest');
                    Velocity_Index(isnan(Velocity_Index)) = 51;

                    % ---- Initialize HD neurons ----
                    hHD = zeros(N_HD_Neurons, 1);
                    rHD = zeros(N_HD_Neurons, T);
                    decodedHD = zeros(T, 1);

                    new_Wrot_matrix = Wrot;

                    for t = 1:T
                        isStationary = checkRanges(stat_startIndices, stat_endIndices, t);

                        K_Inhibitory = K_Inhibitory_base;

                        if opts.shuffleDuringStationary && isStationary
                            K_Excitatory = (K_Exc_gain + 17) * K_Recurrent;
                        else
                            K_Excitatory = K_Excitatory_base;
                        end

                        % Visual input (hill centered at Az(t))
                        Iv0 = Iv_norm;
                        Iv0 = Iv0 / max(Iv0) * K_Visual;
                        shiftBins = round(Az(t) * N_HD_Neurons / 360);
                        Iv = circshift(Iv0, shiftBins);

                        if t == 1
                            Ii = zeros(N_HD_Neurons, 1);
                            Irec = zeros(N_HD_Neurons, 1);
                        else
                            Ii = sum(K_Inhibitory * rHD(:, t-1));
                        end

                        Wrot_effective = Wrot; 

                        useShuffle = opts.shuffleDuringStationary && isStationary && (t > 1) && ...
                                     strcmpi(opts.shuffleMode, 'head_restrained');

                        if useShuffle
                            Wrot_effective = shuffleWrotFraction(Wrot, opts.shuffleFraction);
                            new_Wrot_matrix = Wrot_effective;
                        end

                        % Recurrent input using effective matrix
                        if t > 1
                            Irec = K_Excitatory * Wrot_effective(:, :, Velocity_Index(t)) * rHD(:, t-1);
                        end

                        % Update dynamics
                        hHD = exp(-1/tau) * hHD + (1/tau) * (Iv + (Irec - Ii) * RECCURENT_GAIN);
                        rHD(:, t) = (1 + exp(-2*beta*(hHD - alpha))).^(-1);

                        % Decode
                        a = sum([rHD(:,t).*cos(HD_Neurons_PD')  rHD(:,t).*sin(HD_Neurons_PD')]) / sum(rHD(:,t));
                        decodedHD(t) = atan2d(a(2), a(1));

                        % Live plot
                        if opts.plotEvery > 0 && mod(t, opts.plotEvery) == 1
                            subplot(211); cla; hold on
                            plot(HD_Neurons_PD*180/pi, rHD(:,t), '.r');
                            plot(HD_Neurons_PD*180/pi, Iv / K_Visual, '.k');
                            axis([-180 180 0 1]);
                            set(gca,'XTick',-180:45:180);
                            xlabel('HD (°)');
                            ylabel('a.u.');
                            legend('HD Neurons firing','Direct visual input','Location','Eastoutside');

                            subplot(212); cla; hold on
                            plot(time(1:t), decodedHD(1:t), 'or');
                            plot(time(1:t), mod(Az(1:t)+180,360)-180);
                            legend('Ring HD Signal','Actual HD','Location','Eastoutside');

                            pause(0.001);
                        end
                    end

                    % ---- Save ----
                    fname = strcat( ...
                        "Inh_gain", num2str(K_Inh_gain), ...
                        "Exc_gain", num2str(K_Exc_gain), ...
                        "bimodal_vis_inp", num2str(false), ...
                        "gaus_ratio", num2str(gaus_ratio), ...
                        "visual_std", num2str(Visual_std), ...
                        opts.fileSuffix);

                    fullpath = fullfile(opts.saveDir, fname);

                    save(fullpath, ...
                        "hHD","rHD","decodedHD","Az", ...
                        "stat_startIndices","stat_endIndices", ...
                        "Wrot","new_Wrot_matrix", ...
                        "K_Inh_gain","K_Exc_gain","gaus_ratio","Visual_std","opts");
                end
            end
        end
    end
end

% ---------------- helpers ----------------

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end

function Wshuf = shuffleWrotFraction(Wrot, frac)
    frac = max(0, min(1, frac));
    Wshuf = Wrot;
    if frac == 0
        return;
    end

    [m,n,p] = size(Wrot);
    total = m*n*p;
    numToShuffle = round(frac * total);

    idx = randperm(total, numToShuffle);

    flat = reshape(Wrot, 1, []);
    vals = flat(idx);
    vals = vals(randperm(numel(vals)));
    flat(idx) = vals;

    Wshuf = reshape(flat, m, n, p);
end
