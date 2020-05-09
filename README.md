# Chart recognition and analyses with python
Project to perform ML and AI operations under image to perform difficult tasks such as recognize and extract a graph from image, predict future of the graph, save it locally.
There is no point to be sad as we are going to develop a great design and mobile optimization for the program!!!

Input and Output data:<br>
Input:<br>
Any image of popular format(better .png or .lpg!)<br>

Output data:<br>
Processed image files that differs from user desire. It can include: points.csv, where X nad Y data of the graph is written, reveal.png, where those points are plotted including linear or polynomial regression, also we can get .npy data format for furher scientific process, but it's optional, of course. <br>

Here you can see structure of program:<br>
<pre>
root:
--docs
  --app.py
      main file that launches the program. Uses kivy as main library
    --check_chart.py
        modeule to check if given photo is a chart. IMPORTANT! Needs a model that cant't be loaded in GitHub,<br> so it's a good for you to create your own. 
     --crop_black_background.py
        short module with some functions to make photo look more Graph-ish
    --test.kv
        file written in kivy language to provide fast) and good interface
    --grah_adt.py
        Contains class GraphADT for working with a graph points
  
  --images
    --alt images that change in process
  
  --models
    --model_class.json
        realy easy, don't bother it
    --IMPORTANT MODEL THAT CANT BE LOADED!
         to differ chart from another image 
  other unimportant temp docs<br>
</pre>

Using app will be easu for you as all buttons say it for you :

<img src="https://github.com/Paliy2/sites/blob/master/img/beta.jpg">

Even now you are able to differ any graph from human or dog or anything else. We have cosy and simple interface, so you can enjoy now. Just download files and stay home)
