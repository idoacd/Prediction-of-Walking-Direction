# Prediction-of-Walking-Direction
In this repository we predict the direction/next step a person takes. The prediction is based on the use of a Long-Short-Term-Memory model which estimates the probability function of some unknown walking process.

How and why can we assume next step could be predicted? 
We assume that given a trajectory (and of course the end position of the trajectory), the next step is drawn from an unknown probability function. Given that, and based on collected data, the predictor needs to predict the most likely next step.

In this repository you can find a naive generator of data. E.g, walking trajectories in a made-up supermarket. Structure-wise, a supermarket could be much like a maze. However, in a supermarket people spend time next to display stands, or get back to some points when they forgot something.
