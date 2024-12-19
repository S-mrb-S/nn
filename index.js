const tf = require('@tensorflow/tfjs-node');

// داده‌های آموزشی
const trainingData = [
  "زندگی زیباست.",
  "دوست داشتن خوب است.",
  "امید همیشه وجود دارد.",
  "خوشحالی در دل است.",
  "دوستی ارزشمند است."
];

// پیش‌پردازش داده‌ها
const preprocessData = (data) => {
  return data.map(sentence => sentence.toLowerCase().split(" "));
};

const processedData = preprocessData(trainingData);

// ایجاد دیکشنری از کلمات
const wordSet = new Set();
processedData.forEach(sentence => {
  sentence.forEach(word => wordSet.add(word));
});

const wordList = Array.from(wordSet);
const wordIndex = {};
wordList.forEach((word, index) => {
  wordIndex[word] = index;
});

// کدگذاری جملات
const encodeSentences = (sentences) => {
  return sentences.map(sentence => {
    return sentence.map(word => wordIndex[word]);
  });
};

const encodedData = encodeSentences(processedData);

// ایجاد مدل
const model = tf.sequential();
model.add(tf.layers.embedding({ inputDim: wordList.length, outputDim: 8, inputLength: 5 }));
model.add(tf.layers.lstm({ units: 128 }));
model.add(tf.layers.dense({ units: wordList.length, activation: 'softmax' }));
model.compile({ optimizer: 'adam', loss: 'sparseCategoricalCrossentropy' });

// آموزش مدل
const trainModel = async () => {
  const xs = tf.tensor2d(
    encodedData.map(sentence => {
      const paddedSentence = sentence.slice(0, 5);
      while (paddedSentence.length < 5) {
        paddedSentence.push(0);
      }
      return paddedSentence;
    })
  );

  const ys = tf.tensor1d(
    encodedData.map(sentence => {
      return sentence.length > 1 ? sentence[sentence.length - 1] : 0; // بررسی طول جمله
    }),
    'float32'
  );

  await model.fit(xs, ys, {
    epochs: 100,
    batchSize: 1,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        console.log(`Epoch ${epoch}: loss = ${logs.loss}`);
      },
    },
  });
};

// تولید متن
const generateText = (input) => {
    let generated = [...input];
  
    // اطمینان از اینکه ورودی 5 کلمه دارد
    while (generated.length < 5) {
      generated.unshift(0); // اضافه کردن 0 به ابتدای آرایه
    }
  
    for (let i = 0; i < 5; i++) {
      const inputTensor = tf.tensor2d([generated.slice(-5)]);
      const prediction = model.predict(inputTensor);
      const nextWordIndex = prediction.argMax(-1).dataSync()[0];
  
      // بررسی ایندکس قبل از اضافه کردن
      if (nextWordIndex >= 0 && nextWordIndex < wordList.length) {
        generated.push(nextWordIndex);
      } else {
        console.error("ایندکس نامعتبر:", nextWordIndex);
        break; // یا می‌توانید یک مقدار پیش‌فرض اضافه کنید
      }
    }
  
    return generated.map(index => wordList[index]).join(" ");
  };
  

// اجرای برنامه
(async () => {
  await trainModel();
  const inputSentence = ["زندگی", "زیباست"];
  const generatedText = generateText(inputSentence.map(word => wordIndex[word]));
  console.log("متن تولید شده:", generatedText);
})();
