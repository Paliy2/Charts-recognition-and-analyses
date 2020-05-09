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
root:<br>
--docs<br>
  --app.py<br>
      main file that launches the program. Uses kivy as main library<br>
    --check_chart.py<br>
        modeule to check if given photo is a chart. IMPORTANT! Needs a model that cant't be loaded in GitHub, so it's a good for you to create your own. <br>
     --crop_black_background.py<br>
        short module with some functions to make photo look more Graph-ish<br>
    --test.kv<br>
        file written in kivy language to provide fast) and good interface<br>
    --grah_adt.py<br>
        Contains class GraphADT for working with a graph points<br>
  
  --images<br>
    --alt images that change in process<br>
  
  --models<br>
    --model_class.json<br>
        realy easy, don't bother it<br>
    --IMPORTANT MODEL THAT CANT BE LOADED!<br>
         to differ chart from another image <br>
  other unimportant temp docs<br>
</pre>
Even now you are able to differ any graph from human or dog or anything else. We have cosy and simple interface, so you can enjoy now. Just download files and stay home)
