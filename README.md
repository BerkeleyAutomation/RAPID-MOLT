## RAPID-MOLT: A Meso-scale, Open-source, Low-cost Testbed for Robot Assisted Precision Irrigation and Delivery

To study the automation of plant-level precision irrigation, we present a modular, open-source testbed that enables real-time, fine-grained data collection and irrigation actuation. RAPID-MOLT costs USD $610 and is sized to fit a small laboratory floor space of 0.37 m^2. The functionality of the platform is evaluated by measuring the correlation between plant growth (Leaf Area Index) and water stress (Crop Water Stress Index) with irrigation volume. In line with biological studies, the observed plant growth is positively correlated with irrigation volume while water stress is negatively correlated. This testbed is the basis for ongoing work on learning-based irrigation controllers. 

<img src="/img/FullAssembly.png" alt="drawing" class="centerImage" width="300" />

## Platform Basics
The contributions are as follows:
1. A hardware system design that enables both plant level sensing with multispectral imaging and also precise irrigation control with Raspberry Pi actuated solenoids.
2. A software infrastructure and data processing pipeline to evaluate irrigation controllers, including the automatic extraction of Leaf Area Index and Crop Water Stress Index from RGB and thermal images.
3. Experimental data from two experiments with different crops, reflecting plant growth and water stress data in response to different irrigation schedules.

<img src="/img/FlowChart.png" alt="drawing" class="centerImage" width="300" />

As seen in the flow chart above, each platform unit functions together as follows. Plant and environmental images and data are collected by the sensing unit, and sent to the cloud processing unit for the creation of data-specific irrigation schedules. These schedules are relayed to the irrigation unit for irrigation deployment to the potted plants.

CAD models for construction, code for the software infrastructure, and experimental data can all be found on this repository. 

