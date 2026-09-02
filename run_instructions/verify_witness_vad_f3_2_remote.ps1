$ErrorActionPreference = "Stop"

$repository = (Resolve-Path ".").Path
$remote = "root@connect.cqa1.seetacloud.com"
$port = 19423
$remoteRepository = "/root/autodl-tmp/vadmy_code"

git -C $repository push origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    throw "git push failed"
}

$remoteCommand = @"
set -e
cd $remoteRepository
source /etc/network_turbo
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsanet
bash run_instructions/run_witness_vad_f3_2_dsanet.sh --clean
"@

ssh -p $port $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote DSANet verification failed"
}

$metricCommand = @"
cd $remoteRepository
python -c 'import json; print(json.load(open("../vadmy_data/witness_vad/dsanet/f3_2_signed_support/target_margin.json"))["target_margin_pp"])'
"@
$metric = ssh -p $port $remote $metricCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote metric read failed"
}
@($metric)[-1]
