# TSED - Commodities

Time-Series Enriched with Domain-specific terms to forecasting the daily future price of agricultural commodities.

# Method

The proposed method, a representation of time-series combined with features extracted from a vector representation of texts. Below illustrates the steps performed in the method.

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2215016122001388-ga1_lrg.jpg" width="600px" alt="table2"/>
</p>

# Experimental Configuration

The cross-validation for time-series was used to evaluate the proposed model in the experimental evaluation. The first training step was performed with 30% of the data (F1), and at each cross-validation iteration, a day is added to the training to predict the next step ahead.

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2215016122001388-gr3_lrg.jpg" width="600px" alt="table2"/>
</p>

The variable y′ in Equation 8 represents the forecast of commodity prices h days ahead, and n represents approximately 1230 forecasts (daily) performed in the test stage.

# Results

Figure below shows the graph of the true and forecasted values of commodities with forecasting horizon h = 1. The red and blue points represent the days when the forecast reached the MAPE equal to zero. 

<p align="center">
  <img src="https://ars.els-cdn.com/content/image/1-s2.0-S2215016122001388-gr4_lrg.jpg" width="900px" alt="table2"/>
</p>

More details of the results available [here](https://colab.research.google.com/drive/1zhXSrMmOtN3eSgCZfel-K9mhrH9153zA?usp=sharing).

# Citation

To cite TSED in your work, please use the following bibtex reference:

```
@article{dos2022enrichment,
  title={On the enrichment of time series with textual data for forecasting agricultural commodity prices},
  author={dos Reis Filho, Ivan Jos{\'e} and Marcacini, Ricardo Marcondes and Rezende, Solange Oliveira},
  journal={MethodsX},
  pages={101758},
  year={2022},
  publisher={Elsevier}
}
```
More details of the paper available [here](https://www.sciencedirect.com/science/article/pii/S2215016122001388).

# Reference

Time Series representation models Enriched with text Domain resources (TSED)-[Repository](https://github.com/ivanfilhoreis/tsed)
