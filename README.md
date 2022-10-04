# Function approximation

This is a work that aims at approximating 4 functions using Neural Networks. 
To visualize the project in jupyter, run the next commands :

Pull the docker image
```bash
 docker pull cheikh42/new_project 
```
Run the docker image
```bash
 docker run -it -p 8888:8888 cheikh42/new_project  
```
## The functions 

- <img src="https://latex.codecogs.com/svg.image?f_0(x):=x^2,&space;&space;x&space;\in&space;[-10,&space;10]" title="f_0(x):=x^2, x \in [-10, 10]" />
- <img src="https://latex.codecogs.com/svg.image?f_1(x):=xx-yy,&space;(x,y)&space;\in&space;\Re^2" title="f_1(x):=xx-yy, (x,y) \in \Re^2" />
- <img src="https://latex.codecogs.com/svg.image?f_2(x):=xye^{(-xx&space;-&space;yy)},(x,y)&space;\in&space;[-4,4]" title="f_2(x):=xye^{(-xx - yy)},(x,y) \in [-4,4]" />
- <img src="https://latex.codecogs.com/svg.image?f_3(x):=sin(x),\)&space;&space;\(&space;x\in&space;\Re" title="f_3(x):=sin(x), ( x\in \Re" ) />

To visual this functions, view the notebook ``` functions_visualisation.ipynb ```.

## The datasets
The datasets are generated using ```data_generator.py ``` and stored in the repository ```datasets ```.

## Two Neural networs

The Feed-forward models used are classes generated in ```feed_forward.py ```. And the Recurrent model is generated in ``` recurrent.py ```.

## Training and figures

The training is done by calling the functions in ```training.py ```. And the figures are stored in the repository ``` figures.py ```.
