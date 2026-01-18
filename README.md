# Fischertechnik Multiprocess Department-Conveyor Health Model

# Cognitive DT experiment settings

## Experiments datasets: 
- The dataset selected for the experiment is the [NASA Bearing Dataset (Kaggle)](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset?select=3rd_test).
- The selected dataset is the _Set No. 2_ (see the dataset repository for further details).

## Experiment Scenario

The experimental scenario is the 
[Fischertechnik Multiprocess Station with Oven](https://www.fischertechnik.de/en/products/industry-and-universities/training-models/536632-multi-working-station-with-furnace-24v). 
The selected scenario is considered as a whole department constituted by 5 machines, namely: 
(i) the Oven station, (ii) the Vacuum Gripper Carrier station, (iii) the Turntable station, 
(iv) the Saw station, and (v) the Conveyor station. 

Each machine is controlled by a 24V Industrial Soft-PLC 
([RevolutionPi](https://revolutionpi.com/en/products/revpi-core)) connected via MQTT to a 
remote server. The server collects the data and hosts one Digital Twin (DT) for each machine, thus 
obtaining an Oven DT, a Vacuum Gripper Carrier DT, and so on. A higher level composed DT has always
been implemented, fed by machine-level DTs, thus gaining observability capabilities about the whole 
department state.The Cyber-Physical System (CPS) resulting from the collaboration between the 
physical Conveyor and its DT is the object of the experiment.

Based on the selected dataset ,an Autoencoder health state classification model has been built and 
embedded in the Conveyor DT, thus augmenting the computation capabilities and intelligence level of 
the physical Conveyor *PUT CIT. Then, each data point of the collected dataset has been streamed 
from a dummy Physical Adapter towards the Conveyor DT simulating the streaming of data coming from 
the physical domain. The streaming of data and the Conveyor operations were started synchronously. 
As a result, from the Conveyor DT view point, received vibrational data belonged to the physical 
Conveyor. 

When the embedded model detects a degradation in the Conveyor bearings health state, it can react in 
different ways, such as sending a simple notification or building up a more structured reaction 
strategy. For the experiment purpose, a reaction strategy involving the whole department has been 
implemented.

---

## Dataset key information
- The Kaggle dataset tracks 4 bearings vibration data
- Each data point is gathered every 10 minutes
- Total entries are 982 (EVALUATE TO CHANGE TO 984)
- Assuming that each entry is triggered every second, an experiment can run for (982 s)/(60 s/min) = 16,37 min
- Given the model, the MAE is calculated for every point with respect to the actual prevision 
- If the MAE exceeds a certain threshold, a bearing can be considered broken. 

## Experiment setup: 
- Running tim from 10 to 20 min; 
- The Conveyor has been set to run continuously, stopping only when a piece reaches its 
light-barrier; 
- During the run, the vibration and classification output has been logged into a .csv file; 
- The set print output should also be saved into a .txt file for each run.

## Logged information
- Conveyor DT
  - Received vibrational data (vib-sensor.csv - only for conveyor)
  - Interaction file that contains generated events and received actions requests 
  - Speed file - logged at each change

- Other machines DT: 
  - Interaction file that contains generated events and received actions requests 

- Department DT: 
  - Interaction file that contains generated events and received actions requests

### DUBBI DA CHIARIRE: 
- ma io lo uso il modello in queto progetto? Se no, cancellalo

---

# Experiment result

The implemented and logged breakdown strategy is depicted in the following image: 

![Breakdown strategy](output/graph/breakdown-strategy-timeline.png)

- a breakdown threshold has been set (following the dataset content and the AutoEncoder model);
- At machine breakdown, the Conveyor DT generates anomaly event (1);
- The anomaly event is propagated and received by the Department DT;
- The Department DT sends stop request (2) to the broken Conveyor DT (3);
- Machine DT starts the stopping procedure by lowering its speed until the last workpiece leaves 
the department (4);
- When the last work piece leaves the Conveyor, a notification is sent to the Department DT (5);
- A stop request is sent by the Department DT to all the machines (6) stopping the whole department.


