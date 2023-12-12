# EdgeFlex

### Packages
- `pip3 install vector-quantize-pytorch`
  - `torch.distributed.is_available() and torch.distributed.is_initialized()`
- `wget https://download.pytorch.org/models/mobilenet_v2-b0353104.pth`

### Accuracy Profiling
- Modify batch file `hpc.sh` to train models with diverse quantization settings
- Profiled results are stored at `env/accuracy.pkl`

### Latency Profiling
- Evaluate model inference latency `test_latency.sh` on AGX, NX, TX2 embedded devices
- Profiled results are stored at `env/latency.pkl`

### DRL
- Agent defined in `agent/`
- Environment defined in `env/`
- `python train_agent.py --sla SLA --beta BETA`
  - Or Use batch file `agent_hpc.sh`
- `python test_agent.py --sla SLA --beta BETA`
  - Remember to change PATH/TO/PRETRAINED/MODEL
- Results are in folder `result/`