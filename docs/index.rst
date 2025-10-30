Welcome to ML-Dash
====================

**ML-Dash** is a lightweight Python SDK for ML experiment metricing and data storage.

Metric your machine learning experiments with zero setup. Start locally on your laptop, then seamlessly scale to a remote server when you need team collaboration. No configuration files, no complex setup - just clean, intuitive code.

**Key highlights:**

- **Zero setup** - Start metricing in 60 seconds with local filesystem storage
- **Dual modes** - Work offline (local) or collaborate (remote server)
- **Fluent API** - Intuitive builder pattern for logs, parameters, metrics, and files

Installation
------------

.. code-block:: bash

   pip install ml_dash

Quick Example
-------------

.. code-block:: python
   :linenos:

   from ml_dash import Experiment

   with Experiment(name="my-experiment", project="my-project", local_path=".ml_dash") as experiment:
       # Log messages
       experiment.log("Training started")

       # Metric parameters
       experiment.parameters().set(learning_rate=0.001, batch_size=32)

       # Metric metrics
       experiment.metric("loss").append(value=0.5, epoch=1)

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   overview
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   experiments
   logging
   parameters
   metrics
   files

.. toctree::
   :maxdepth: 2
   :caption: Examples
   :hidden:

   basic-training
   hyperparameter-search
   model-comparison

