source /usr/local/Ascend/cann-9.0.0/set_env.sh
cd /home/u2501112130/MLSys-HW
export PYTHONPATH="$PWD/tokenspeed/python:$PWD/tokenspeed/tokenspeed-kernel/python:$PWD/tokenspeed/tokenspeed-kernel-npu/python"
export ASCEND_RT_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
exec timeout --kill-after=15s 600s .venv/bin/python -m pytest -q -rs -p no:cacheprovider tokenspeed/tokenspeed-kernel-npu/test/test_gdn.py tokenspeed/tokenspeed-kernel-npu/test/test_causal_conv.py tokenspeed/tokenspeed-kernel-npu/test/test_gated_rmsnorm.py tokenspeed/tokenspeed-kernel-npu/test/test_gemma_layer.py tokenspeed/tokenspeed-kernel-npu/test/test_mha_d256.py tokenspeed/test/runtime/test_qwen35_npu_initialization.py
