# Repository for the paper "Ultrafast Traffic Nowcasting and Control via Differentiable Agent-based Simulation"
## About
This repository contains the code and processed data used in the paper
["Ultrafast Traffic Nowcasting and Control via Differentiable Agent-based Simulation"
](https://arxiv.org/abs/2603.25068).
---
## Requirement
The code was tested with the following package versions:
- Python>=3.10
- numpy>=1.24,<2.0
- pandas>=2.0
- scipy>=1.10
- jax>=0.4.30
- jaxlib>=0.4.30
- optax>=0.2.0
- jaxopt>=0.8.0
- matplotlib>=3.7
---
## Repository Structure
- `src/`
  Core implementation of the differentiable agent-based traffic simulator and the joint calibration driver.
- `scripts/`
  Entry-point scripts for calibration.
- `examples/`
  Example shell scripts for launching calibration runs.
- `data/`
  Directory for placing the input TNTP network files (not bundled).
---
## Datasets
The road-network data is not bundled with this repository. Download the TNTP files
from the [Transportation Networks for Research](https://github.com/bstabler/TransportationNetworks)
project and place them under:
- `data/chicagosketch/ChicagoSketch_net.tntp`, `ChicagoSketch_node.tntp`
- `data/SiouxFalls/SiouxFalls_net.tntp`, `SiouxFalls_node.tntp`
---
## License
See the LICENSE file for details.
---
## Citation

```
@misc{makinoshima2026ultrafasttrafficnowcastingcontrol,
      title={Ultra-fast Traffic Nowcasting and Control via Differentiable Agent-based Simulation}, 
      author={Fumiyasu Makinoshima and Yuya Yamaguchi and Eigo Segawa and Koichiro Niinuma and Sean Qian},
      year={2026},
      eprint={2603.25068},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2603.25068}, 
}
```
