# Pensieve PPO

## About Pensive-PPO

This is an easy TensorFlow implementation of Pensieve[1]. 
In detail, we trained Pensieve via PPO rather than A3C.
It's a stable version, which has already prepared the training set and the test set, and you can run the repo easily: just type

```
python train.py
```

instead. Results will be evaluated on the test set (from HSDPA) every 300 epochs.

## Experimental Results

We reported the training curve of entropy weight beta, reward, and entropy respectively. Results were evaluated over the Oboe network traces.

Tips: the orange curve: pensieve-ppo; the blue curve: pensieve-a2c
