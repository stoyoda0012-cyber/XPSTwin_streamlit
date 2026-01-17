# XPS_IRF_Simulator

XPS (X-ray Photoelectron Spectroscopy) IRF Simulator & Resolution Explorer with advanced deconvolution analysis.

## Features

- **2D Source & Detector Modeling**: Realistic simulation of X-ray source spot profiles and detector geometry
- **Fermi Edge Fitting**: High-precision fitting using Fermi-Dirac + Gaussian convolution with global optimization
- **IRF Parameter Estimation**: Reverse-engineering of instrumental response function from observed spectra
- **Resolution Analysis**: Theoretical vs experimental resolution comparison with component breakdown
- **Interactive UI**: Real-time parameter adjustment with Streamlit

## Quick Start

### Local Installation

```bash
git clone https://github.com/stoyoda0012-cyber/XPSTwin_streamlit.git
cd XPSTwin_streamlit
pip install -r requirements.txt
streamlit run app.py
```

### Online Demo

Visit the live app: [XPS_IRF_Simulator](https://xpstwin.streamlit.app)

### Documentation

- [Mathematical Foundation](https://stoyoda0012-cyber.github.io/XPSTwin_streamlit/XPS_IRF_Simulator_Mathematical_Foundation.html) - Detailed mathematical basis and implementation notes

## Usage

1. **Adjust Parameters**: Use the sidebar to control:
   - Temperature (K)
   - Detector parameters (Kappa, Theta, Resolution)
   - Source parameters (Spot size, Asymmetry, Rotation)
   - Energy gradient (Alpha)
   - Noise levels (Poisson & Gaussian)

2. **Run Analysis**:
   - **Fermi Edge Fitting**: Extract Ef shift and total resolution
   - **IRF Parameter Estimation**: Reverse-engineer geometric parameters

3. **Visualize Results**:
   - 1D spectra with noise
   - 2D spot profiles
   - Instrumental response functions
   - Fitting residuals and quality metrics

## Technical Details

### Simulation Engine

- **Energy Range**: -100 to +100 meV around Fermi level
- **Grid Resolution**: 500-1000 points
- **Physics**: Fermi-Dirac statistics with thermal broadening
- **Convolution**: Gaussian IRF with variable resolution

### Fitting Algorithms

- **Two-Stage Optimization**:
  1. Differential Evolution (global search)
  2. Levenberg-Marquardt (local refinement)
- **Parameters Fitted**: Ef shift, σ_total, Temperature, Amplitude, Offset
- **Quality Metrics**: R², Residuals, Covariance matrix

### IRF Components

- Detector intrinsic resolution
- Smile curvature (Kappa)
- Detector tilt (Theta)
- Source size (X, Y)
- Energy gradient (Alpha)
- Spot asymmetry (Gamma_X, Gamma_Y, Rotation)

## Project Structure

```
XPSTwin_streamlit/
├── app.py                          # Main Streamlit application
├── xps_twin/
│   ├── core/
│   │   ├── grid.py                 # Energy grid and axis management
│   │   └── physics.py              # Fermi-Dirac and physical models
│   ├── components/
│   │   ├── source.py               # X-ray source 2D modeling
│   │   └── analyzer_2d.py          # Detector 2D projection
│   ├── models/
│   │   └── twin_engine.py          # Main simulation engine
│   └── analysis/
│       ├── deconvolution.py        # Fermi edge fitting & IRF estimation
│       └── optimizer.py            # Legacy optimizer
├── requirements.txt
└── README.md
```

## Dependencies

- Python 3.8+
- streamlit
- numpy
- scipy
- matplotlib
- pandas

## Citation

If you use this simulator in your research, please cite:

```
XPS_IRF_Simulator & Resolution Explorer
https://github.com/stoyoda0012-cyber/XPSTwin_streamlit
```

## License

MIT License

## Author

Satoshi Toyoda

## Acknowledgments

Built with Claude Code (Anthropic)
