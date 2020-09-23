# nyc_taxi
## Data Science Take Home Test
Every month New York City releases data about trips taken in NYC taxis. This data can be found here: (NYC Taxi & Limousine Commission - Trip Record Data) and is described by the data dictionary (https://www1.nyc.gov/assets/tlc/downloads/pdf/data_dictionary_trip_records_yellow.pdf)

A new ride sharing company is your client. They are looking to start up a rival service to Uber or Lyft in NYC and are interested in using the Taxi Data to get insight into the transportation sector in New York. They have tasked you to get to know the data and start to build some simple models off of it.

In the following tasks, you should focus on yellow cab data from the months of March, June, and November in 2017. Feel free to use more than that, but those months should be the minimum.

Feel free to use any programming language and libraries of your choosing. Results should be delivered as a either a GitHub repo with the code and results (preferred), a zipped directory of code and results, or Colab Notebook(s). Note: If you post your work online, please ensure there are no references to CARTO or links to this homework.

You should spend approximately 4 hours on these tasks. Try to document your thought process as much as possible.

You have three tasks:

### ~~Data exploration and cleaning~~

* ~~Download and assess the data~~
* ~~Identify and document any issues with the data~~
* ~~Document how you resolved these issues~~

### Data Summary

Create a brief report that summarizes the core features of the data and calls out interesting trends, etc. This can be in the form of a notebook-like document (Jupyter, RMarkdown, etc.) or a slide deck. The audience for this should be other data scientists who need to understand that dataset.

### Model Building

The client is launching a new ride sharing program in New York similar to Uber or Lyft. At the end of each trip they want their app to suggest a tip amount to the rider. The company has not acquired any of their own data yet, so they have tasked you with producing a model based off of the taxi data. This model should predict the likely tip amount for a trip based on the other trip attributes. You can assume that the ride sharing company can provide data that has the same attributes as the taxi data for each trip.

In building the model, consider the following requirements:

* The model should be built from the taxi dataset. You can supplement the taxi data with external datasets, but this is not a requirement.
* Document your choice of model / algorithm, discussing why you chose it over alternatives.
    * Document how you assess your models performance.
    * Discuss any limitations or caveats of the model which might be an issue in implementing it.
    * Discuss how you might improve your model going forward.
* Discuss how you might turn this model in to an API the company can use.