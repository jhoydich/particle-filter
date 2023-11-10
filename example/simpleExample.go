package main

import (
	"fmt"
	"math"
	particlefilter "particle-filter"

	"gonum.org/v1/gonum/stat/distuv"
)

func main() {
	noiseGenerator := distuv.Normal{Mu: 0, Sigma: .05}

	pf := particlefilter.CreatePF(500, 1, 0, 10, 0, 10, .1, .1, math.Pi/16)

	p := particlefilter.Particle{X: 1.0, Y: 1.0, Heading: math.Pi / 2, Weight: 1}

	for i := 0; i < 20; i++ {
		r := SimpleReading{
			X: p.X + noiseGenerator.Rand(),
			Y: p.Y + noiseGenerator.Rand(),
		}

		pf.CalculateWeights(r)
		pf.ResampleAndFuzz()
		fmt.Println("Before Move: ", "Particle:", p.X, p.Y, p.Heading, "Filter:", pf.EstimatedX, pf.EstimatedY, pf.EstimatedHeading)
		p.Move(.5+noiseGenerator.Rand(), 0)

		pf.MoveParticles(.5, 0)
		fmt.Println("After Move: ", "Particle:", p.X, p.Y, p.Heading, "Filter:", pf.EstimatedX, pf.EstimatedY, pf.EstimatedHeading)
	}

	//pf.CalculateWeights()
}

type SimpleReading struct {
	X float64
	Y float64
}

func (s SimpleReading) CalculateWeight(p particlefilter.Particle) float64 {

	dist := math.Sqrt(math.Pow((s.X-p.X), 2) + math.Pow((s.Y-p.Y), 2))

	return particlefilter.CalculateNormDist(dist, 0, .1)

}
