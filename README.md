# Minimalist Reinforcement Learning (MRL)



A transparent, modular collection of reinforcement learning algorithms and tools designed for research and rapid prototyping.



## Motivation



This project is heavily inspired by [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3). While SB3 is the industry standard, I found that its high level of abstraction sometimes acted as a "black box," making it difficult to experiment with the core mechanics of an algorithm.



I am building this repository to:



* **Deepen Understanding:** Manually implementing RL mechanics to truly "experience" the nuances of the math.

* **Prioritize Customization:** Create a codebase where it's easy to swap out network architectures (like Transformers or VAEs) without fighting a complex framework.

* **Keep it Simple:** A "no-frills" approach to RL research that I can easily integrate into my future projects.



## Installation



You can install this library directly from GitHub to use in your own environments:



```bash

pip install git+https://github.com/username/repository_name.git



```



## Current Roadmap



This is a living research project. It is currently in an early alpha stage, and I’m adding features as my research requires them.



### **Core Stability & Optimization**



* [ ] **Gradient Clipping:** Implement global and local norm clipping.

* [ ] **Weight Initialization:** Standardize Orthogonal and Xavier initialization methods.

* [ ] **Vectorized Environments:** Parallelizing agents to speed up data collection.



### **Architectures & Features**



* [ ] **Transformer Backbones:** Moving beyond standard MLPs for sequential decision making.

* [ ] **Preprocessing Suite:** Integrated tools for state normalization and Autoencoders (VAE/AE).

* [ ] **Monitoring:** Integration for real-time visualization and logging.



### **Future Explorations**



* [ ] Support for additional RL algorithms (SAC, TD3, etc.).

* [ ] Potential migration to JAX for faster hardware acceleration.



---



## Disclaimer



This is primarily a personal tool for my own research and development. While I aim for it to be functional and clean, it is a work in progress and may change significantly over time. Feel free to explore the code, but use it at your own risk!

