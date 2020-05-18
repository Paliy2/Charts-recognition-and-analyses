# Chart recognition and analyses with python
Project to perform ML and AI operations under image to perform difficult tasks such as recognize and extract a graph from an image, predict future of the graph or save it locally.
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
     --crop_black_background.py
        short module with some functions to make photo look more Graph-ish
    --test.kv
        file written in kivy language to provide fast) and good interface
    --grah_adt.py
        Contains class GraphADT for working with a graph points
    --image_adt
    --quantize
    --process
    
  --images
    --alt images that change in process
    
  other unimportant temp docs<br>
</pre>

Using app will be easy for you as all buttons say it for you, even now you are able to interact with some beta-interface :

<img src="https://github.com/Paliy2/sites/blob/master/img/beta.jpg">

To check the realization, just save your data into 2D numpy array and "feed" it to the GraphADT.

Also We are going to make it more usable for everyone, so peple can use our algorithms to photoGraph their everyday life. For example, here you can see image of a pencil:

<a href='https://youtu.be/3gukhl7fFc0' target="_blank"><img src="https://github.com/Paliy2/sites/blob/master/img/pencil.jpg"></a>

and how the app sees it now: 

<a href='https://youtu.be/3gukhl7fFc0' target="_blank"><img src="https://github.com/Paliy2/sites/blob/master/img/pencil_res.png"></a>


Differnece isn't great, but all functionality we nned to add is to remove the noise from picture and build the graph. Then we plan to extract the formula of an item, so you can save the whole world data easily just in one row!

Yeah, now it looks terrible, here you see digitized form of the pencil above:

<a href='https://youtu.be/3gukhl7fFc0' target="_blank"><img src="https://github.com/Paliy2/sites/blob/master/img/reveal.png"></a>

In the close future we are going to finish this project, so you can use it for personal purposes. 

But even now you are able to graph form of human or dog or anything else. We have cosy, fast and simple interface, so you can enjoy now. Just download files and stay home)


Check out our awesome video by this link
<a href='https://youtu.be/3gukhl7fFc0' target="_blank">PhotoGraph</a>
