document.getElementById('carCount').value = localStorage.getItem("carCount") || 1;
document.getElementById('mutationAmount').value = localStorage.getItem("mutationAmount") || '0.5';

const carCanvas = document.getElementById("carCanvas");
carCanvas.width = 200;
const networkCanvas = document.getElementById("networkCanvas");
networkCanvas.width = 500;

const carCtx = carCanvas.getContext("2d");
const networkCtx = networkCanvas.getContext("2d");

const road = new Road(carCanvas.width / 2, carCanvas.width * 0.9);

const N = Number(document.getElementById('carCount').value);
const cars = generateCars(N);
let bestCar = cars[0];

if (localStorage.getItem("bestBrain")) {
  for (let i = 0; i < cars.length; i++) {
    cars[i].brain = JSON.parse(localStorage.getItem("bestBrain"));
    if (i !== 0) {
      NeuralNetwork.mutate(cars[i].brain, Number(document.getElementById('mutationAmount').value));
    }
  }
}

const traffic = [
  new Car(road.getLaneCenter(1), -100, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(0), -300, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(2), -300, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(1), -500, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(2), -500, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(0), -700, 30, 50, "DUMMY", 2, getRandomColor()),
  new Car(road.getLaneCenter(2), -700, 30, 50, "DUMMY", 2, getRandomColor())
];

animate();

function save() {
  localStorage.setItem("bestBrain", JSON.stringify(bestCar.brain));
}

function discard() {
  localStorage.removeItem("bestBrain");
}

function generateCars(N) {
  const cars = [];
  for (let i = 1; i <= N; i++) {
    cars.push(new Car(road.getLaneCenter(1), 100, 30, 50, "AI"));
  }
  return cars;
}

function animate(time) {
  for (let i = 0; i < traffic.length; i++) {
    traffic[i].update(road, []);
  }
  
  for (let i = 0; i < cars.length; i++) {
    cars[i].update(road, traffic);
  }
  bestCar = cars.find(c => c.fitness === Math.max(...cars.map(c => c.fitness)));

  carCanvas.height = window.innerHeight;
  networkCanvas.height = window.innerHeight;

  carCtx.save();
  carCtx.translate(0, -bestCar.y + carCanvas.height * 0.7);

  road.draw(carCtx);
  for (let i = 0; i < traffic.length; i++) {
    traffic[i].draw(carCtx);
  }
  carCtx.globalAlpha = 0.2;
  for (let i = 0; i < cars.length; i++) {
    cars[i].draw(carCtx);
  }
  carCtx.globalAlpha = 1;
  bestCar.draw(carCtx, true);

  carCtx.restore();

  networkCtx.lineDashOffset = -time / 50;
  Visualizer.drawNetwork(networkCtx, bestCar.brain);

  regenerateTraffic();
  requestAnimationFrame(animate);
}

function regenerateTraffic() {
  console.log("Regenerating Traffic");
  console.log("Traffic Y:", traffic[traffic.length - 1].y);
  console.log("Cars Y:", cars[0].y);

  if (traffic[traffic.length - 1].y > cars[0].y) {
    for (let i = 0; i < 5; i++) {
      const laneIndex = randomize(0, 2);
      const yPos = cars[0].y - randomize(carCanvas.height - 340, carCanvas.height + 500);
      const color = getRandomColor();
  
      traffic.push(new Car(road.getLaneCenter(laneIndex), yPos, 30, 50, "DUMMY", 2, color));
    }
  }
}
