#!/usr/bin/env bash
set -euo pipefail

profile="${1:-train-large}"

case "$profile" in
  smoke)
    req_gpus=0
    req_mem_gb=32
    req_cpus=4
    ;;
  preprocess|export-yolo)
    req_gpus=0
    req_mem_gb=180
    req_cpus=24
    ;;
  train)
    req_gpus=1
    req_mem_gb=180
    req_cpus=24
    ;;
  train-large|h100)
    req_gpus=1
    req_mem_gb=320
    req_cpus=32
    ;;
  multi-gpu)
    req_gpus=4
    req_mem_gb=700
    req_cpus=96
    ;;
  cpu-highmem)
    req_gpus=0
    req_mem_gb=700
    req_cpus=64
    ;;
  *)
    echo "Usage: $0 [smoke|preprocess|export-yolo|train|train-large|h100|multi-gpu|cpu-highmem]" >&2
    exit 2
    ;;
esac

if ! command -v freenodes >/dev/null 2>&1; then
  echo "freenodes is not available on this host. Run this on the HPC login node." >&2
  exit 1
fi

freenodes -cg | awk \
  -v profile="$profile" \
  -v req_gpus="$req_gpus" \
  -v req_mem_gb="$req_mem_gb" \
  -v req_cpus="$req_cpus" '
function mem_free_gb(mem_field, parts, value) {
  split(mem_field, parts, "/")
  value = parts[1]
  gsub(/gb/, "", value)
  return value + 0
}
function gpu_free_count(gpu_field, parts) {
  if (gpu_field == "--") return 0
  split(gpu_field, parts, "/")
  return parts[1] + 0
}
function cpu_free_count(cpu_field, parts) {
  split(cpu_field, parts, "/")
  return parts[1] + 0
}
function gpu_rank(gpu) {
  gpu = tolower(gpu)
  if (gpu == "h100") return 1000
  if (gpu == "l40s") return 850
  if (gpu == "a100") return 800
  if (gpu == "a40") return 650
  if (gpu == "l4") return 450
  if (gpu == "a10") return 400
  return 0
}
NR == 1 { next }
{
  node = $1
  queues = $2
  free_mem = mem_free_gb($3)
  free_cpu = cpu_free_count($5)
  gpu = $6
  free_gpu = gpu_free_count($7)

  if (free_mem < req_mem_gb || free_cpu < req_cpus) next
  if (req_gpus > 0 && (gpu == "--" || free_gpu < req_gpus)) next
  if (req_gpus == 0 && profile != "cpu-highmem" && gpu != "--") next
  if (profile == "cpu-highmem" && gpu != "--") next

  score = free_mem + free_cpu
  if (req_gpus > 0) score += gpu_rank(gpu) + (30 * free_gpu)
  if (profile == "train" && queues ~ /gpus/) score += 1200
  if ((profile == "train-large" || profile == "h100" || profile == "multi-gpu") && gpu == "h100") score += 500
  if (profile == "cpu-highmem" && queues ~ /bigmem/) score += 300

  queue = "preemptible"
  if (profile == "train" && queues ~ /gpus/) queue = "gpus"
  if (profile == "cpu-highmem" && queues ~ /bigmem/) queue = "bigmem"

  printf "%010.1f\t%s\t%s\t%s\t%d\t%d\t%d\n", score, node, queue, gpu, free_gpu, free_mem, free_cpu
}' | sort -r | head -1 | awk -F '\t' \
  -v req_gpus="$req_gpus" \
  -v req_mem_gb="$req_mem_gb" \
  -v req_cpus="$req_cpus" \
  -v profile="$profile" '
NF == 0 {
  print "No matching free node found for profile: " profile
  exit 1
}
{
  node = $2
  queue = $3
  gpu = $4
  free_gpu = $5
  free_mem = $6
  free_cpu = $7
  print "profile=" profile
  print "recommended_node=" node
  print "recommended_queue=" queue
  print "gpu_type=" gpu
  print "free_gpus=" free_gpu
  print "free_mem_gb=" free_mem
  print "free_cpus=" free_cpu
  printf "#PBS -q %s\n", queue
  if (req_gpus > 0) {
    printf "#PBS -l select=1:mem=%dgb:ncpus=%d:ngpus=%d:host=%s\n", req_mem_gb, req_cpus, req_gpus, node
  } else {
    printf "#PBS -l select=1:mem=%dgb:ncpus=%d:host=%s\n", req_mem_gb, req_cpus, node
  }
}'
