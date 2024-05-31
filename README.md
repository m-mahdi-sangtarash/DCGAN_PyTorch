# DCGAN Model with PyTorch 

DCGAN (Deep Convolutional Generative Adversarial Network) is a type of GAN (Generative Adversarial Network) that utilizes convolutional and convolutional-transpose layers to generate more realistic images.


## Components of DCGAN

* Generator

  ![DCGAN model](readme-imgs/DCGAN.png)
  
* Discriminator

  ![DCGAN model](readme-imgs/Discriminator.jpg)


## Dataset

To train the model in this practice project, I used the "CelebA" dataset. You can see examples of the images in this dataset below.

_Sample images from CelebA Dataset_

![sample train images](Outputs/smaple-train-images.png)


### Outputs

* Gen and Disc training loss plot

  ![plot](Outputs/Gen-disc-training-loss.png)


* Animation Output

  <link rel="stylesheet"
href="https://maxcdn.bootstrapcdn.com/font-awesome/4.4.0/css/font-awesome.min.css">
<script language="javascript">
  function isInternetExplorer() {
    ua = navigator.userAgent;
    /* MSIE used to detect old browsers and Trident used to newer ones*/
    return ua.indexOf("MSIE ") > -1 || ua.indexOf("Trident/") > -1;
  }

  /* Define the Animation class */
  function Animation(frames, img_id, slider_id, interval, loop_select_id){
    this.img_id = img_id;
    this.slider_id = slider_id;
    this.loop_select_id = loop_select_id;
    this.interval = interval;
    this.current_frame = 0;
    this.direction = 0;
    this.timer = null;
    this.frames = new Array(frames.length);

    for (var i=0; i<frames.length; i++)
    {
     this.frames[i] = new Image();
     this.frames[i].src = frames[i];
    }
    var slider = document.getElementById(this.slider_id);
    slider.max = this.frames.length - 1;
    if (isInternetExplorer()) {
        // switch from oninput to onchange because IE <= 11 does not conform
        // with W3C specification. It ignores oninput and onchange behaves
        // like oninput. In contrast, Microsoft Edge behaves correctly.
        slider.setAttribute('onchange', slider.getAttribute('oninput'));
        slider.setAttribute('oninput', null);
    }
    this.set_frame(this.current_frame);
  }

  Animation.prototype.get_loop_state = function(){
    var button_group = document[this.loop_select_id].state;
    for (var i = 0; i < button_group.length; i++) {
        var button = button_group[i];
        if (button.checked) {
            return button.value;
        }
    }
    return undefined;
  }

  Animation.prototype.set_frame = function(frame){
    this.current_frame = frame;
    document.getElementById(this.img_id).src =
            this.frames[this.current_frame].src;
    document.getElementById(this.slider_id).value = this.current_frame;
  }

  Animation.prototype.next_frame = function()
  {
    this.set_frame(Math.min(this.frames.length - 1, this.current_frame + 1));
  }

  Animation.prototype.previous_frame = function()
  {
    this.set_frame(Math.max(0, this.current_frame - 1));
  }

  Animation.prototype.first_frame = function()
  {
    this.set_frame(0);
  }

  Animation.prototype.last_frame = function()
  {
    this.set_frame(this.frames.length - 1);
  }

  Animation.prototype.slower = function()
  {
    this.interval /= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.faster = function()
  {
    this.interval *= 0.7;
    if(this.direction > 0){this.play_animation();}
    else if(this.direction < 0){this.reverse_animation();}
  }

  Animation.prototype.anim_step_forward = function()
  {
    this.current_frame += 1;
    if(this.current_frame < this.frames.length){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.first_frame();
      }else if(loop_state == "reflect"){
        this.last_frame();
        this.reverse_animation();
      }else{
        this.pause_animation();
        this.last_frame();
      }
    }
  }

  Animation.prototype.anim_step_reverse = function()
  {
    this.current_frame -= 1;
    if(this.current_frame >= 0){
      this.set_frame(this.current_frame);
    }else{
      var loop_state = this.get_loop_state();
      if(loop_state == "loop"){
        this.last_frame();
      }else if(loop_state == "reflect"){
        this.first_frame();
        this.play_animation();
      }else{
        this.pause_animation();
        this.first_frame();
      }
    }
  }

  Animation.prototype.pause_animation = function()
  {
    this.direction = 0;
    if (this.timer){
      clearInterval(this.timer);
      this.timer = null;
    }
  }

  Animation.prototype.play_animation = function()
  {
    this.pause_animation();
    this.direction = 1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_forward();
    }, this.interval);
  }

  Animation.prototype.reverse_animation = function()
  {
    this.pause_animation();
    this.direction = -1;
    var t = this;
    if (!this.timer) this.timer = setInterval(function() {
        t.anim_step_reverse();
    }, this.interval);
  }
</script>

<style>
.animation {
    display: inline-block;
    text-align: center;
}
input[type=range].anim-slider {
    width: 374px;
    margin-left: auto;
    margin-right: auto;
}
.anim-buttons {
    margin: 8px 0px;
}
.anim-buttons button {
    padding: 0;
    width: 36px;
}
.anim-state label {
    margin-right: 8px;
}
.anim-state input {
    margin: 0;
    vertical-align: middle;
}
</style>

<div class="animation">
  <img id="_anim_imgd675d8220ca64f12bcb3b6a6487e344b">
  <div class="anim-controls">
    <input id="_anim_sliderd675d8220ca64f12bcb3b6a6487e344b" type="range" class="anim-slider"
           name="points" min="0" max="1" step="1" value="0"
           oninput="animd675d8220ca64f12bcb3b6a6487e344b.set_frame(parseInt(this.value));">
    <div class="anim-buttons">
      <button title="Decrease speed" aria-label="Decrease speed" onclick="animd675d8220ca64f12bcb3b6a6487e344b.slower()">
          <i class="fa fa-minus"></i></button>
      <button title="First frame" aria-label="First frame" onclick="animd675d8220ca64f12bcb3b6a6487e344b.first_frame()">
        <i class="fa fa-fast-backward"></i></button>
      <button title="Previous frame" aria-label="Previous frame" onclick="animd675d8220ca64f12bcb3b6a6487e344b.previous_frame()">
          <i class="fa fa-step-backward"></i></button>
      <button title="Play backwards" aria-label="Play backwards" onclick="animd675d8220ca64f12bcb3b6a6487e344b.reverse_animation()">
          <i class="fa fa-play fa-flip-horizontal"></i></button>
      <button title="Pause" aria-label="Pause" onclick="animd675d8220ca64f12bcb3b6a6487e344b.pause_animation()">
          <i class="fa fa-pause"></i></button>
      <button title="Play" aria-label="Play" onclick="animd675d8220ca64f12bcb3b6a6487e344b.play_animation()">
          <i class="fa fa-play"></i></button>
      <button title="Next frame" aria-label="Next frame" onclick="animd675d8220ca64f12bcb3b6a6487e344b.next_frame()">
          <i class="fa fa-step-forward"></i></button>
      <button title="Last frame" aria-label="Last frame" onclick="animd675d8220ca64f12bcb3b6a6487e344b.last_frame()">
          <i class="fa fa-fast-forward"></i></button>
      <button title="Increase speed" aria-label="Increase speed" onclick="animd675d8220ca64f12bcb3b6a6487e344b.faster()">
          <i class="fa fa-plus"></i></button>
    </div>
    <form title="Repetition mode" aria-label="Repetition mode" action="#n" name="_anim_loop_selectd675d8220ca64f12bcb3b6a6487e344b"
          class="anim-state">
      <input type="radio" name="state" value="once" id="_anim_radio1_d675d8220ca64f12bcb3b6a6487e344b"
             >
      <label for="_anim_radio1_d675d8220ca64f12bcb3b6a6487e344b">Once</label>
      <input type="radio" name="state" value="loop" id="_anim_radio2_d675d8220ca64f12bcb3b6a6487e344b"
             checked>
      <label for="_anim_radio2_d675d8220ca64f12bcb3b6a6487e344b">Loop</label>
      <input type="radio" name="state" value="reflect" id="_anim_radio3_d675d8220ca64f12bcb3b6a6487e344b"
             >
      <label for="_anim_radio3_d675d8220ca64f12bcb3b6a6487e344b">Reflect</label>
    </form>
  </div>
</div>


<script language="javascript">
  /* Instantiate the Animation class. */
  /* The IDs given should match those used in the template above. */
  (function() {
    var img_id = "_anim_imgd675d8220ca64f12bcb3b6a6487e344b";
    var slider_id = "_anim_sliderd675d8220ca64f12bcb3b6a6487e344b";
    var loop_select_id = "_anim_loop_selectd675d8220ca64f12bcb3b6a6487e344b";
    var frames = new Array(11);
    
  for (var i=0; i<11; i++){
    frames[i] = "animation_output_frames/frame" + ("0000000" + i).slice(-7) +
                ".png";
  }


    /* set a timeout to make sure all the above elements are created before
       the object is initialized. */
    setTimeout(function() {
        animd675d8220ca64f12bcb3b6a6487e344b = new Animation(frames, img_id, slider_id, 1000.0,
                                 loop_select_id);
    }, 0);
  })()
</script>
