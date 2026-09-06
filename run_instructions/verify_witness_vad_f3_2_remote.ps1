param(
    [ValidateSet("ucf", "xd", "all")]
    [string]$Dataset = "all"
)

$ErrorActionPreference = "Stop"

$repository = (Resolve-Path ".").Path
$remote = "root@connect.cqa1.seetacloud.com"
$port = 19423
$remoteRepository = "/root/autodl-tmp/vadmy_code"

git -C $repository push origin HEAD:main
if ($LASTEXITCODE -ne 0) {
    throw "git push failed"
}

$datasetList = if ($Dataset -eq "all") { "ucf xd" } else { $Dataset }
$remoteCommand = @"
set -e
cd $remoteRepository
source /etc/network_turbo
git pull --ff-only origin main
source /root/miniconda3/etc/profile.d/conda.sh
conda activate dsanet
bash run_instructions/verify_witness_vad_f3_2_remote.sh '$datasetList'
"@

ssh -p $port $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote DSANet verification failed"
}
