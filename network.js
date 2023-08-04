const trainingData = [
    { inputs: [0, 0], outputs: [0] },
    { inputs: [0, 1], outputs: [1] },
    { inputs: [1, 0], outputs: [1] },
    { inputs: [1, 1], outputs: [0] }
  ];
  
  class NeuralNetwork {
    constructor(neuronCounts) {
      this.levels = [];
      for (let i = 0; i < neuronCounts.length - 1; i++) {
        this.levels.push(new Level(neuronCounts[i], neuronCounts[i + 1]));
      }
    }
  
    static feedForward(givenInputs, network) {
      let outputs = Level.feedForward(givenInputs, network.levels[0]);
      for (let i = 1; i < network.levels.length; i++) {
        outputs = Level.feedForward(outputs, network.levels[i]);
      }
      return outputs;
    }
  
    static mutate(network, amount = 1) {
      network.levels.forEach((level) => {
        for (let i = 0; i < level.biases.length; i++) {
          level.biases[i] = lerp(level.biases[i], Math.random() * 2 - 1, amount);
        }
        for (let i = 0; i < level.weights.length; i++) {
          for (let j = 0; j < level.weights[i].length; j++) {
            level.weights[i][j] = lerp(
              level.weights[i][j],
              Math.random() * 2 - 1,
              amount
            );
          }
        }
      });
    }
  
    train(trainingData, learningRate = 0.3, iterations = 1000) {
      for (let iter = 0; iter < iterations; iter++) {
        for (let data of trainingData) {
          const inputs = data.inputs;
          const expectedOutputs = data.outputs;
  
          // Perform feedforward
          const outputs = NeuralNetwork.feedForward(inputs, this);
  
          // Perform backpropagation
          let error = 0;
          for (let i = this.levels.length - 1; i >= 0; i--) {
            const level = this.levels[i];
            if (i === this.levels.length - 1) {
              // Output layer
              for (let j = 0; j < level.outputs.length; j++) {
                const output = level.outputs[j];
                const expectedOutput = expectedOutputs[j];
                const delta = expectedOutput - output;
                error += Math.abs(delta);
                level.errors[j] = delta * output * (1 - output);
              }
            } else {
              // Hidden layers
              for (let j = 0; j < level.outputs.length; j++) {
                const output = level.outputs[j];
                let sum = 0;
                for (let k = 0; k < this.levels[i + 1].outputs.length; k++) {
                  const nextLevel = this.levels[i + 1];
                  sum += nextLevel.weights[j][k] * nextLevel.errors[k];
                }
                level.errors[j] = sum * output * (1 - output);
              }
            }
          }
  
          // Update weights and biases using gradient descent
          for (let i = 0; i < this.levels.length; i++) {
            const level = this.levels[i];
            for (let j = 0; j < level.inputs.length; j++) {
              for (let k = 0; k < level.outputs.length; k++) {
                const delta =
                  learningRate * level.errors[k] * level.inputs[j];
                level.weights[j][k] += delta;
              }
            }
            for (let j = 0; j < level.outputs.length; j++) {
              const delta = learningRate * level.errors[j];
              level.biases[j] += delta;
            }
          }
        }
      }
    }
  }
  
  class Level {
    constructor(inputCount, outputCount) {
      this.inputs = new Array(inputCount);
      this.outputs = new Array(outputCount);
      this.errors = new Array(outputCount);
      this.biases = new Array(outputCount);
  
      this.weights = [];
      for (let i = 0; i < inputCount; i++) {
        this.weights[i] = new Array(outputCount);
      }
  
      Level.#randomize(this);
    }
  
    static #randomize(level) {
      for (let i = 0; i < level.inputs.length; i++) {
        for (let j = 0; j < level.outputs.length; j++) {
          level.weights[i][j] = Math.random() * 2 - 1;
        }
      }
  
      for (let i = 0; i < level.biases.length; i++) {
        level.biases[i] = Math.random() * 2 - 1;
      }
    }
  
    static feedForward(givenInputs, level) {
      for (let i = 0; i < level.inputs.length; i++) {
        level.inputs[i] = givenInputs[i];
      }
  
      for (let i = 0; i < level.outputs.length; i++) {
        let sum = 0;
        for (let j = 0; j < level.inputs.length; j++) {
          sum += level.inputs[j] * level.weights[j][i];
        }
  
        if (sum > level.biases[i]) {
          level.outputs[i] = 1;
        } else {
          level.outputs[i] = 0;
        }
      }
  
      return level.outputs;
    }
  }
  