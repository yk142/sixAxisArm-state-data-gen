function sa_gen_pulse_el(shardI, shardN)
%SA_GEN_PULSE_EL 弾性つきデータのうち、パルス励振（高トルク。短くて速い）だけを分担生成する（issue #73）。
repo = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(repo,'config'), genpath(fullfile(repo,'src')));
if ischar(shardI), shardI = str2double(shardI); end
if ischar(shardN), shardN = str2double(shardN); end
S = [saScenarioSetPayload(6,'full',1,[1 2 4 6 8]), saScenarioSetPulse(6,120,12,[0 1 2 4 6 8])];
nm = {S(strcmp({S.pattern},'pulse')).name};
genSaDataset(struct('nAx',6,'preset','full','replicas',1,'pulse',[120 12],'names',{nm},'shard',[shardI shardN],'cacheOnly',true,'elastic',struct()));
end
