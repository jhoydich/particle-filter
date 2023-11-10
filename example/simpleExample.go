package main

import (
	"fmt"
	"math"

	pf "github.com/jhoydich/particle-filter"
	"gonum.org/v1/gonum/stat/distuv"
)

func main() {
	noiseGenerator := distuv.Normal{Mu: 0, Sigma: .05}

	filter := pf.CreatePF(500, 1, 0, 10, 0, 10, .1, .1, math.Pi/16)

	p := pf.Particle{X: 1.0, Y: 1.0, Heading: math.Pi / 2, Weight: 1}

	for i := 0; i < 20; i++ {
		r := SimpleReading{
			X: p.X + noiseGenerator.Rand(),
			Y: p.Y + noiseGenerator.Rand(),
		}

		filter.CalculateWeights(r)
		filter.ResampleAndFuzz()
		fmt.Println("Before Move: ", "Particle:", p.X, p.Y, p.Heading, "Filter:", filter.EstimatedX, filter.EstimatedY, filter.EstimatedHeading)
		p.Move(.5+noiseGenerator.Rand(), 0)

		filter.MoveParticles(.5, 0)
		fmt.Println("After Move: ", "Particle:", p.X, p.Y, p.Heading, "Filter:", filter.EstimatedX, filter.EstimatedY, filter.EstimatedHeading)
	}

	//pf.CalculateWeights()
}

type SimpleReading struct {
	X float64
	Y float64
}

func (s SimpleReading) CalculateWeight(p pf.Particle) float64 {

	dist := math.Sqrt(math.Pow((s.X-p.X), 2) + math.Pow((s.Y-p.Y), 2))

	return pf.CalculateNormDist(dist, 0, .1)

}
