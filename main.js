import "@babel/polyfill";
import * as tf from '@tensorflow/tfjs';
import * as mobilenetModule from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier';

const TOPK = 10;
const IMAGE_SIZE = 369;
const NUMBER_OF_CLASSES = 2;

class Main {
  constructor() {
    this.infoTexts = [];
    this.training = -1;
    this.videoPlaying = false;

    this.bindPage();

    this.video = document.createElement('video');
    this.video.setAttribute('autoplay', '');
    this.video.setAttribute('playsinline', '');

    document.body.appendChild(this.video);

    const classesContentContainer = document.createElement("div");

    classesContentContainer.className = "classes-content-container";

    document.body.appendChild(classesContentContainer);

    for (let i = 0; i < NUMBER_OF_CLASSES; i++) {
      const div = document.createElement('div');
      div.className = "buttons-container";

      const button = document.createElement('button')
      button.innerText = `Treinar ${i + 1}º categoria`;

      button.addEventListener('mouseup', () => this.training = -1);
      button.addEventListener('mousedown', () => this.training = i);
      button.addEventListener("touchend", () => this.training = -1);
      button.addEventListener("touchstart", () => this.training = i);

      const infoText = document.createElement('p');
      infoText.innerText = " Nenhum exemplo da categoria fornecido";
      div.appendChild(infoText);
      div.appendChild(button);
      this.infoTexts.push(infoText);

      classesContentContainer.appendChild(div);
    }

    navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false
    })
      .then((stream) => {
        this.video.srcObject = stream;
        this.video.width = IMAGE_SIZE;
        this.video.height = IMAGE_SIZE - 92;

        this.video.addEventListener('playing', () => this.videoPlaying = true);
        this.video.addEventListener('paused', () => this.videoPlaying = false);
      });
  }

  async bindPage() {
    this.tf = tf;
    this.knn = knnClassifier.create();
    this.mobilenet = await mobilenetModule.load();

    this.start();
  }

  start() {
    if (this.timer) this.stop();

    this.video.play();
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }

  stop() {
    this.video.pause();

    cancelAnimationFrame(this.timer);
  }

  async animate() {
    if (this.videoPlaying) {
      const image = tf.fromPixels(this.video);

      let logits;
      const infer = () => this.mobilenet.infer(image, 'conv_preds');

      if (this.training != -1) {
        logits = infer();

        this.knn.addExample(logits, this.training);
      }

      const numClasses = this.knn.getNumClasses();
      if (numClasses > 0) {

        logits = infer();
        const res = await this.knn.predictClass(logits, TOPK);

        for (let i = 0; i < NUMBER_OF_CLASSES; i++) {

          const exampleCount = this.knn.getClassExampleCount();

          if (exampleCount[i] > 0) {
            this.infoTexts[i].innerText = ` ${exampleCount[i]} exemplos fornecidos | Precisão: ${(res.confidences[i] * 100).toFixed(1)}%`
          }
        }
      }

      image.dispose();

      if (logits != null) logits.dispose();
    }
    this.timer = requestAnimationFrame(this.animate.bind(this));
  }
  
  train(className) {
    const imagesMaxIndex = className === "leanwork" ? 130 : 141;

    for (let index = 0; index < imagesMaxIndex; index += 1) {
      this._loadImageAndClassify(index, className);
    }
  }

  export() {}

  loadImageAndClassify(index, className) {
    const img = document.createElement("img");

    img.src = `./images/${className}/${index}.jpg`;

    img.style.display = "none";

    const that = this;

    img.onload = () => {
      const fromPixels = tf.fromPixels(img);

      let logits;

      const infer = () => that.mobilenet.infer(fromPixels, 'conv_preds');

      logits = infer();

      that.knn.addExample(logits, className === "leanwork" ? 0 : 1);

      img.remove();
    }

    document.body.appendChild(img);
  }
}

window.addEventListener('load', () => new Main());
