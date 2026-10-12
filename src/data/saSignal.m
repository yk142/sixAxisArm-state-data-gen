function u = saSignal(type, tVec, amp, opts)
%SASIGNAL 励振用のトルク信号を生成する（8 kHz の時刻列 tVec 上、1 軸分）。
%   u = SASIGNAL(type, tVec, amp, opts)
%     type  'bln'    バンドリミテッドノイズ（opts.fMax [Hz]、opts.seed）
%           'chirp'  対数周波数掃引（opts.f0, opts.f1 [Hz]、任意で opts.phiDeg 初期位相 [deg]）
%           'step'   ステップ（opts.tOn [s]、opts.sign = ±1）
%     amp   振幅 [N*m]（信号の最大絶対値が amp になる）。u は [N x 1]
%   BLN は 1 kHz で生成して 4 次 Butterworth（filtfilt、零位相）で帯域制限し、pchip で tVec に補間する。
%   最大絶対値で正規化するので、実効値は amp より小さい（ピーク係数は約 3〜4）。軸ごとに seed を変えて独立にする。
tVec = tVec(:);
switch type
    case 'bln'
        assert(isfield(opts,'fMax') && isfield(opts,'seed'), 'saSignal:opts', 'bln には fMax と seed が必要です');
        fsG = 1000;
        nG = max(ceil(tVec(end) - tVec(1))*fsG + 1, 64);
        tG = tVec(1) + (0:nG-1)'/fsG;
        rs = RandStream('twister','Seed',opts.seed);
        [b, a] = butter(4, opts.fMax/(fsG/2));
        y = filtfilt(b, a, randn(rs, nG, 1));
        y = y / max(abs(y));
        u = amp * interp1(tG, y, tVec, 'pchip', 'extrap');
        u = max(min(u, amp), -amp);
    case 'chirp'
        phi = 0;  if isfield(opts,'phiDeg'), phi = opts.phiDeg; end
        t = tVec - tVec(1);
        u = amp * chirp(t, opts.f0, t(end), opts.f1, 'logarithmic', phi);
    case 'step'
        sg = 1;  if isfield(opts,'sign'), sg = opts.sign; end
        u = amp * sg * double(tVec >= opts.tOn);
    otherwise
        error('saSignal:type', '未知の type: %s', type);
end
end
