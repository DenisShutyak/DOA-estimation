# DOA-estimation

This is a project for comparing various algorithms for DoA estimation in case of multiple signals and trying to solve DoA issue for 2 singnls with convolutional neural network. Here i compare Capons method, Thermal noise and MUSIC algorithms applied to real antenna array. Real antenn array means that array elements may be missplced a bit, there can be amplitude distribution on elements and directional diagram of single element may not be a circle.
# Directional diagram of array
Directional diagrm of real antenna array is given bu formula:

<a href="https://www.codecogs.com/eqnedit.php?latex=S(\theta)&space;=&space;\sum_n&space;S_n(\theta)&space;AD_n(\theta)&space;exp((2&space;\pi&space;d&space;sin\theta)/\lambda)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?S(\theta)&space;=&space;\sum_n&space;S_n(\theta)&space;AD_n(\theta)&space;exp((2&space;\pi&space;d&space;sin\theta)/\lambda)" title="S(\theta) = \sum_n S_n(\theta) AD_n(\theta) exp((2 \pi d sin\theta)/\lambda)" /></a>

where S<sub>n</sub> is directional diagramm of a single element,  AD is amplitude distribution, d is array of positions of elements, <a href="https://www.codecogs.com/eqnedit.php?latex=\lambda" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\lambda" title="\lambda" /></a> is lengthwave. For more [info](https://books.google.ru/books/about/Digital_Spectral_Analysis.html?id=uEOjngEACAAJ&redir_esc=y) and [overview](https://en.wikipedia.org/wiki/Antenna_array).


In this project i decided to use cosine amplitude distribution:

<a href="https://www.codecogs.com/eqnedit.php?latex=AD(\theta)&space;=&space;delta&space;-&space;1&space;&plus;&space;cos(d&space;N&space;sin(\theta))^n" target="_blank"><img src="https://latex.codecogs.com/gif.latex?AD(\theta)&space;=&space;delta&space;-&space;1&space;&plus;&space;cos(d&space;N&space;sin(\theta))^n" title="AD(\theta) = delta - 1 + cos(d N sin(\theta))^n" /></a>

And array elements are missplaced from equidistant condition for ~5 lengthwaves.

# Difference between real and ideal directional diagram

Here you can see directional diagram of idela 10-element antenna array
![ideal_dd](https://user-images.githubusercontent.com/73283847/131486603-416cf551-21fb-450b-a8e0-77bd9fb8542c.png)

Adding amplitude distribution with delta = 1 and raising the cosine to the power 1 and 2 leads to a significant decrease in the level on the sides of the center
![delta1power1dd](https://user-images.githubusercontent.com/73283847/131486545-46198d76-e33c-4b5d-a8ab-0372a4ef9084.png)
![delta1power2dd](https://user-images.githubusercontent.com/73283847/131486552-95afcef1-82cb-4f5a-a448-f2861ccf0d3a.png)

Adding random error to elements positions leads to significant distortion
![3lambda](https://user-images.githubusercontent.com/73283847/131486556-a2b88a2e-2ad0-4916-b858-30780079ef93.png)
![5lambda](https://user-images.githubusercontent.com/73283847/131486558-6b229ba5-52e0-4499-83fc-b0a10f8ccd1a.png)

This difference can be used to improve the performance of DoA algorithms.

# DoA methods

Direction-of-arrival (DOA) estimation refers to the process of retrieving the direction information of several electromagnetic waves/sources from the outputs of a number of receiving antennas that form a sensor array. DOA estimation is a major problem in array signal processing and has wide applications in radar, sonar, wireless communications, etc.
I decided to try Capons method, Thermal Noise algorithm and MUSIC methods because they are widely used and well researched. Moreover they perform really well.  
More [info about DoA](https://www.sciencedirect.com/topics/engineering/direction-of-arrival-estimation) and [methods](https://iopscience.iop.org/article/10.1088/1742-6596/1279/1/012012/pdf).

# Comparison of methods

The comparison is based on [Rayleigh criterion](https://en.wikipedia.org/wiki/Angular_resolution#The_Rayleigh_criterion). To compare methods one should set 2 signals close to each other and start moving them to each other. When it becomes impossible to distinguish one signal from another we should stop modeling, usually this is the moment when the signal power and the power in the space between signals is less than 3dB. Then we should calculate, how investigated method outperformed Rayleigh criterion. Then we should increase power of signal and repeat modeling.

Here are results:
1) Ideal antenna array
![идеальная_дн_сверхразрешение](https://user-images.githubusercontent.com/73283847/131495422-b3797bc2-5203-4453-b82e-c72d63040922.png)  
2) Amplitude distribution with delta = 1 and raising the cosine to the power 1
![дельта1степень1разрешение](https://user-images.githubusercontent.com/73283847/131495434-17405c4a-fe60-4453-b92f-59a5c0eaec28.png)  
3) Amplitude distribution with delta = 1 and raising the cosine to the power 2
![дельта1степень2разрешение](https://user-images.githubusercontent.com/73283847/131495436-3d73c428-7538-4e5d-a5ab-1ef3db89780a.png)  
4) Adding random error to elements positions up to 3 lengthwaves
![3длинволн_разрешение](https://user-images.githubusercontent.com/73283847/131495430-6e196098-58d0-46fd-8404-60ea89d05b44.png)  
5) Adding random error to elements positions up to 5 lengthwaves
![5длинволн_разрешение](https://user-images.githubusercontent.com/73283847/131495432-65d589d3-a3d1-4b2d-86a3-a02dfab019b7.png)  

# Neural network

The reason to use a neural network for this task is that classical algorithms can be very slow when simulating large antenna arrays. For exmple Thermal noise method modeling on antenna array with 10 elements (it's not large) and 2 signals arriving takes more than 1 second to estimate DoA on my pc. And this time increases significantly with an increase in the number of elements, so no real time system can be built.
I decided to use a simple convolutional neurel network, here is the graph.
![model](https://user-images.githubusercontent.com/73283847/131499921-cecbae70-67a2-4c2a-9667-7dc2bfa68d9b.png)  

This neural network worked pretty well. Calculation took ~0.05 seconds, and DoA estimation error was 0.003 radians (about 0.17 deg).

![image](https://user-images.githubusercontent.com/73283847/131500549-5f60aa1e-9557-4f13-a4d5-00b7f628a04f.png)

# Conclusion

I do understand that this neural network cannot be aplied for different situation then researched one, but i did it to have some practice using Tensorflow and to use it as refference point for futher research. The rest part of this project seems good to me for understanding DoA estimation procces.
