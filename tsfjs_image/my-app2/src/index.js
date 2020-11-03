import React from "react";
import ReactDOM from "react-dom";
import MagicDropzone from "react-magic-dropzone";
import ReactPlayer from "react-player";

import * as tf from "@tensorflow/tfjs";

import "./styles.css";
const MODEL_URL = process.env.PUBLIC_URL + "/model_web/";
const LABELS_URL = MODEL_URL + "labels.json";
const MODEL_JSON = MODEL_URL + "model.json";

const TFWrapper = (model) => {
  const calculateMaxScores = (scores, numBoxes, numClasses) => {
    const maxes = [];
    const classes = [];
    for (let i = 0; i < numBoxes; i++) {
      let max = Number.MIN_VALUE;
      let index = -1;
      for (let j = 0; j < numClasses; j++) {
        if (scores[i * numClasses + j] > max) {
          max = scores[i * numClasses + j];
          index = j;
        }
      }
      maxes[i] = max;
      classes[i] = index;
    }
    return [maxes, classes];
  };

  const buildDetectedObjects = (
    width,
    height,
    boxes,
    scores,
    indexes,
    classes
  ) => {
    const count = indexes.length;
    const objects = [];
    for (let i = 0; i < count; i++) {
      const bbox = [];
      for (let j = 0; j < 4; j++) {
        bbox[j] = boxes[indexes[i] * 4 + j];
      }
      const minY = bbox[0] * height;
      const minX = bbox[1] * width;
      const maxY = bbox[2] * height;
      const maxX = bbox[3] * width;
      bbox[0] = minX;
      bbox[1] = minY;
      bbox[2] = maxX - minX;
      bbox[3] = maxY - minY;
      objects.push({
        bbox: bbox,
        class: classes[indexes[i]],
        score: scores[indexes[i]],
      });
    }
    return objects;
  };

  const detect = (input) => {
    const batched = tf.tidy(() => {
      const img = tf.browser.fromPixels(input);
      // Reshape to a single-element batch so we can pass it to executeAsync.
      return img.expandDims(0);
    });

    const height = batched.shape[1];
    const width = batched.shape[2];

    return model.executeAsync(batched).then((result) => {
      const scores = result[0].dataSync();
      const boxes = result[1].dataSync();

      // clean the webgl tensors
      batched.dispose();
      tf.dispose(result);

      const [maxScores, classes] = calculateMaxScores(
        scores,
        result[0].shape[1],
        result[0].shape[2]
      );

      const prevBackend = tf.getBackend();
      // run post process in cpu
      tf.setBackend("cpu");
      const indexTensor = tf.tidy(() => {
        const boxes2 = tf.tensor2d(boxes, [
          result[1].shape[1],
          result[1].shape[3],
        ]);
        return tf.image.nonMaxSuppression(
          boxes2,
          maxScores,
          20, // maxNumBoxes
          0.5, // iou_threshold
          0.5 // score_threshold
        );
      });
      const indexes = indexTensor.dataSync();
      indexTensor.dispose();
      // restore previous backend
      tf.setBackend(prevBackend);

      return buildDetectedObjects(
        width,
        height,
        boxes,
        maxScores,
        indexes,
        classes
      );
    });
  };
  return {
    detect: detect,
  };
};

class App extends React.Component {
  state = {
    model: null,
    labels: [],
    preview: "",
    predictions: [],
    isImage: false,
  };
  componentDidMount() {
    tf.loadGraphModel(MODEL_JSON).then((model) => {
      this.setState({
        model: model,
      });
    });
    fetch(LABELS_URL)
      .then((data) => data.json())
      .then((labels) => {
        this.setState({
          labels: labels,
        });
      });
  }

  onDrop = (accepted, rejected, links) => {
    this.setState({ preview: accepted[0].preview || links[0] });
  };

  cropToCanvas = (image, canvas, ctx) => {
    const naturalWidth = image.naturalWidth;
    const naturalHeight = image.naturalHeight;

    canvas.width = image.width;
    canvas.height = image.height;

    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    if (naturalWidth > naturalHeight) {
      ctx.drawImage(
        image,
        (naturalWidth - naturalHeight) / 2,
        0,
        naturalHeight,
        naturalHeight,
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      );
    } else {
      ctx.drawImage(
        image,
        0,
        (naturalHeight - naturalWidth) / 2,
        naturalWidth,
        naturalWidth,
        0,
        0,
        ctx.canvas.width,
        ctx.canvas.height
      );
    }
  };

  onImageChange = (e) => {
    const c = document.getElementById("canvas");
    const ctx = c.getContext("2d");
    this.cropToCanvas(e.target, c, ctx);
    TFWrapper(this.state.model)
      .detect(c)
      .then((predictions) => {
        // Font options.
        const font = "16px sans-serif";
        ctx.font = font;
        ctx.textBaseline = "top";

        predictions.forEach((prediction) => {
          const x = prediction.bbox[0];
          const y = prediction.bbox[1];
          const width = prediction.bbox[2];
          const height = prediction.bbox[3];
          const label = this.state.labels[parseInt(prediction.class)];
          // Draw the bounding box.
          ctx.strokeStyle = "#00FFFF";
          ctx.lineWidth = 4;
          ctx.strokeRect(x, y, width, height);
          // Draw the label background.
          ctx.fillStyle = "#00FFFF";
          const textWidth = ctx.measureText(label).width;
          const textHeight = parseInt(font, 10); // base 10
          //ctx.fillRect(x, y, textWidth + 2, textHeight + 2);
        });

        predictions.forEach((prediction) => {
          const x = prediction.bbox[0];
          const y = prediction.bbox[1];
          const label = this.state.labels[parseInt(prediction.class)];
          // Draw the text last to ensure it's on top.
          ctx.fillStyle = "#000000";
          ctx.fillText(label, x, y - 10);
        });
      });
  };

  onLoadVideo = () => {
    console.log("je suis le video ");
    // ICI LA FONCTION A APPLIQUER AU VIDEO 
  };
  render() {
    return (
      <div className="Dropzone-page">
        {this.state.model ? (
          <MagicDropzone
            className="Dropzone"
            accept="image/jpeg, image/png, .jpg, .jpeg, .png,.mp4"
            multiple={false}
            onDrop={this.onDrop}>
            {this.state.preview ? (
              <ReactPlayer
                onReady={this.onLoadVideo}
                controls={true}
                url={this.state.preview}
              />
            ) : (
              "Choose or drop a file."
            )}
            <canvas id="canvas" />
          </MagicDropzone>
        ) : (
          <div className="Dropzone">Loading model...</div>
        )}
      </div>
    );
  }
}

const rootElement = document.getElementById("root");
ReactDOM.render(<App />, rootElement);
