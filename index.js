const tf = require("@tensorflow/tfjs-node");
const readline = require("readline");

const trainingData = [
  { text: "This movie was great!", label: 1 },
  { text: "This movie was terrible.", label: 0 },
  { text: "I love this book.", label: 1 },
  { text: "This food is really delicious.", label: 1 },
  { text: "This experience was awful.", label: 0 },
  { text: "I absolutely hated this movie.", label: 0 },
  { text: "What a fantastic experience!", label: 1 },
  { text: "I will never watch this again.", label: 0 },
  { text: "This is the best thing ever!", label: 1 },
  { text: "I didn't like it at all.", label: 0 },
];

const preprocessData = (data) => {
  const sentences = data.map((item) => item.text.toLowerCase().split(" "));
  const labels = data.map((item) => item.label);
  return { sentences, labels };
};

const { sentences, labels } = preprocessData(trainingData);

const wordSet = new Set();
sentences.forEach((sentence) => {
  sentence.forEach((word) => wordSet.add(word));
});

const wordList = Array.from(wordSet);
const wordIndex = {};
wordList.forEach((word, index) => {
  wordIndex[word] = index + 1;
});

const encodeSentences = (sentences) => {
  return sentences.map((sentence) => {
    return sentence.map((word) => wordIndex[word] || 0);
  });
};

const encodedData = encodeSentences(sentences);

const maxLength = Math.max(...encodedData.map((sentence) => sentence.length));
const paddedData = encodedData.map((sentence) => {
  const paddedSentence = sentence.slice(0, maxLength);
  while (paddedSentence.length < maxLength) {
    paddedSentence.push(0);
  }
  return paddedSentence;
});

// تغییرات در مدل
const model = tf.sequential();
model.add(
  tf.layers.embedding({
    inputDim: wordList.length + 1,
    outputDim: 16, // افزایش ابعاد خروجی
    inputLength: maxLength,
  })
);
model.add(tf.layers.lstm({ units: 32, returnSequences: false })); // استفاده از LSTM
model.add(tf.layers.dense({ units: 16, activation: "relu" })); // لایه Dense اضافی
model.add(tf.layers.dense({ units: 1, activation: "sigmoid" }));
model.compile({
  optimizer: "adam",
  loss: "binaryCrossentropy",
  metrics: ["accuracy"],
});

const trainModel = async () => {
  const xs = tf.tensor2d(paddedData);
  const ys = tf.tensor1d(labels, "float32");

  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(
          `Epoch ${epoch}: loss = ${logs.loss}, accuracy = ${logs.acc}`
        );
      },
    },
  });
};

const predictSentiment = (text) => {
  const encodedText = encodeSentences([text.toLowerCase().split(" ")]);
  const paddedText = encodedText[0].slice(0, maxLength);
  while (paddedText.length < maxLength) {
    paddedText.push(0);
  }

  const inputTensor = tf.tensor2d([paddedText]);
  const prediction = model.predict(inputTensor);
  const sentiment = prediction.dataSync()[0];

  return sentiment > 0.5 ? "Positive" : "Negative";
};

// تابع برای دریافت ورودی از کاربر
const getUserInput = () => {
  const rl = readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });

  rl.question("Enter a sentence to analyze its sentiment: ", (input) => {
    const sentiment = predictSentiment(input);
    console.log(`Sentence: "${input}" - Sentiment: ${sentiment}`);
    rl.close();
  });
};

(async () => {
  await trainModel();
  getUserInput(); // دریافت ورودی از کاربر
})();
