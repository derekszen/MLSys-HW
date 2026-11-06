source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /home/u2501112130/MLSys-HW
export PYTHONPATH="$PWD/tokenspeed/python:$PWD/tokenspeed/tokenspeed-kernel/python:$PWD/tokenspeed/tokenspeed-kernel-npu/python"
export ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec timeout --kill-after=15s 600s .venv/bin/python reference-hw1/homework-1/scripts/engine_validate.py --physical-device 1 --model models/Qwen3.5-0.8B --reference runs/20261106-reference/model-reference.json --output runs/20261106-engine-validation/validation.json
