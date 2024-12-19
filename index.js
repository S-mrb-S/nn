const tf = require('@tensorflow/tfjs-node');

// داده‌های آموزشی (جملات غمگین)
const trainingData = [
    "زندگی بدون تو بی‌معناست.",
    "غم و اندوه همیشه با من است.",
    "هر روز بدون تو سخت‌تر می‌شود.",
    "دلم برای روزهای خوب گذشته تنگ شده است.",
    "تنهایی من را می‌کشد."
];

// پیش‌پردازش داده‌ها
const preprocessData = (data) => {
    return data.map(sentence => sentence.toLowerCase().split(' '));
};

const processedData = preprocessData(trainingData);

// ایجاد دیکشنری
const wordSet = new Set();
processedData.forEach(sentence => {
    sentence.forEach(word => wordSet.add(word));
});

const wordList = Array.from(wordSet);
const wordIndex = {};
wordList.forEach((word, index) => {
    wordIndex[word] = index;
});

// تبدیل جملات به توکن‌ها
const encodeSentences = (sentences) => {
    return sentences.map(sentence => {
        return sentence.map(word => wordIndex[word]);
    });
};

const encodedData = encodeSentences(processedData);

// ایجاد مدل LSTM
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: wordList.length, outputDim: 8, inputLength: 5 }));
model.add(tf.layers.lstm({ units: 128 }));
model.add(tf.layers.dense({ units: wordList.length, activation: 'softmax' }));
model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy' });

// آموزش مدل
const trainModel = async () => {
    const xs = tf.tensor2d(encodedData.map(sentence => sentence.slice(0, -1)));
    const ys = tf.tensor1d(encodedData.map(sentence => sentence[sentence.length - 1]), 'int32');

    await model.fit(xs, ys, {
        epochs: 100,
        batchSize: 1,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
            }
        }
    });
};

// تولید متن
const generateText = (input) => {
    let generated = input;
    for (let i = 0; i < 5; i++) {
        const inputTensor = tf.tensor2d([generated.slice(-4)]);
        const prediction = model.predict(inputTensor);
        const nextWordIndex = prediction.argMax(-1).dataSync()[0];
        generated.push(nextWordIndex);
    }
    return generated.map(index => wordList[index]).join(' ');
};

// اجرای برنامه
(async () => {
    await trainModel();
    const inputSentence = ["زندگی", "بدون", "تو", "بی‌معناست"];
    const generatedText = generateText(inputSentence.map(word => wordIndex[word]));
    console.log("متن تولید شده:", generatedText);
})();