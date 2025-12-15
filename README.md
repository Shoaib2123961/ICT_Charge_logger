Python-based GUI for calibration and charge measurement using a Bergoz ICT and Libera ADC-500. Communicates via EPICS Channel Access, providing real-time waveform visualization, ADC control, automated offset correction, background subtraction, and continuous charge logging for accelerator diagnostics.

## 🚀 Features

✅ Real-time charge logging and plotting  
✅ EPICS Integration: Real-time communication with the Libera ADC-500 via EPICS Channel Access protocol.
✅ Source selection: Allow selecting a mimic source for testing GUI offline, and  real source in the presence of ADC available over Ethernet. 
✅ Channel selection: Allow selecting ADC input channel with which output of ICT is connected. 
✅ Data acquisition control: Set the number of consecutive triggers to be acquired by setting arm number value. The predefined value is set to be zero for continuous acquisition.   
✅ Attenuation Control: Direct setting of the ADC analog front-end attenuation (10- 31 dB).
✅ Record background: Optional feature to record background charge and subtraction of dark current from next acquisitions.
✅ Start/Stop button: Control over starting and stopping acquisition, and automatically  writing the data to a log file. 
✅ Data Visualization: Dual-axis plotting for live voltage waveform and historical charge tracking.
✅ Vertical cursor: To measure the charge value at any previous time by dragging the cursor horizontally. 
✅ Correction Logic: Incorporation of the algorithms for automatic offset and noise correction.




## 🛠 Requirements

- Python 3.11+
  
Install all dependencies via pip:

```bash
pip install -r requirements.txt
