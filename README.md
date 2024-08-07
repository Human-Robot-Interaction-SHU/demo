# Human-Robot-Interaction-Demo


This demo demonstrates the combination of varius recognation and detection models to enable communication between robot and humans.


## Project Structure:

```
/ProjectRoot
    /Modules (Features)
        /Vision (Feature)
            /Data (Datasets and Preprocessed Data)
            /FacialExpression (Sub-feature)
                - models.py (Model definitions and training scripts)
                - preprocessing.py (Data preprocessing and augmentation)
                - inference.py (Inference and testing scripts)
            /EyeGaze (Sub-feature)
                - models.py
                - preprocessing.py
                - inference.py
            /BodyPose (Sub-feature)
                - models.py
                - preprocessing.py
                - inference.py
        /Sound (Feature)
            /Data (Datasets and Preprocessed Data)
            /VoiceTone (Sub-feature)
                - models.py
                - preprocessing.py
                - inference.py
            /SpeechContent (Sub-feature)
                - models.py
                - preprocessing.py
                - inference.py
    /Integration
        - integration.py (Script for integrating outputs from all modules)
        - visualization.py (Scripts for visualizing the overall state)
    /Utils
        - utils.py (Common utility functions)
```


## How to Run it

To run the script, execute the following command:

-python demo.py


Ensure that all dependencies are installed and the necessary models and weights are available in the specified paths.




