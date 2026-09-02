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
export WITNESS_DATASETS='$datasetList'
config='../vadmy_data/witness_vad/dsanet/f3_2_signed_support/$Dataset/w6/training/config.json'
metric='../vadmy_data/witness_vad/dsanet/f3_2_signed_support/target_margin.json'
recorded_commit=`$(grep -m1 '"git_commit"' "`$config" 2>/dev/null | cut -d'"' -f4 || true)
if [[ -n "`$recorded_commit" ]] && [[ -f "`$metric" ]] && \
   git diff --quiet "`$recorded_commit"..HEAD -- vin_vad run_instructions/run_witness_vad_f3_2_dsanet.sh; then
  echo "reuse matching formal result from `$recorded_commit"
  python - "`$metric" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["target_margin_pp"])
PY
else
  bash run_instructions/run_witness_vad_f3_2_dsanet.sh --clean
fi
"@

ssh -p $port $remote $remoteCommand
if ($LASTEXITCODE -ne 0) {
    throw "remote DSANet verification failed"
}
