---
layout: page
---

<style>
.container-iframe {
    height: 675px;
}

iframe {
    transform: scale(0.5);
    transform-origin: 0 0;
    border-style: solid;
    border-width: 12px;
}
</style>


<h2>The Death Metal Detector</h2>

<blockquote cite="https://en.wikipedia.org/wiki/Death_metal">
Death metal is an extreme subgenre of heavy metal music...played with techniques such as...deep growling vocals; aggressive, powerful drumming, featuring double kick and blast beat techniques... 
<b><u>The lyrical themes of death metal may include slasher film-style violence, political conflict, religion, nature, philosophy, true crime and science fiction.</u></b>

<br>

<a href="https://en.wikipedia.org/w/index.php?title=Death_metal&oldid=1285824176">Wikipedia, Death Metal</a>

</blockquote>


Death Metal lyrics have a certain edge to them above and beyond what Ozzy Osbourne was doing in the 70s. I suspected that Death Metal might use a unique and detectable "language" even when the subject of a song is something other than a horror movie synopsis.

Based on this hypothesis, I built The Death Metal Detector. It's a Complement Naive Bayes machine learning model, 
trained on TF-IDF transformed data on over 200,000 songs. This "scores" the brutality of the lyrics you provide on a scale from 0% to 100%

Try it out below! Copy/paste lyrics from your favorite songs, metal or not, or come up with your own gutteral incantations and see how <b><i>brutal</i></b> they are!

When you're done, read about [how it was made.](pt1-getting-the-data.md)


<div class="container-iframe">
<iframe 
src="https://flask-ui-136635714089.us-west1.run.app/"
width="200%" height="1250" title="Death Metal Detector"
></iframe> 
</div>

[<< Home](../../index.md)<br>
&nbsp;&nbsp;&nbsp;[Tutorial, Pt 1 - Getting The Data >>](pt1-getting-the-data.md)