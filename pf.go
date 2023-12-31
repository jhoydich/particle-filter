package particlefilter

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat/distuv"
)

type Reading interface {
	CalculateWeight(Particle) float64
	Unmarshal(data []byte)
}

type Particle struct {
	X       float64
	Y       float64
	Heading float64
	Weight  float64
}

func (p *Particle) UpdateWeight(weight float64) {
	p.Weight = weight
}

func (p *Particle) Move(dist, ang float64) {
	p.Heading += ang
	if p.Heading < 0 {
		p.Heading += (2.0 * math.Pi)
	} else if p.Heading > 2.0*math.Pi {
		p.Heading -= (2.0 * math.Pi)
	}
	p.X += dist * math.Cos(p.Heading)
	p.Y += dist * math.Sin(p.Heading)
}

type ParticleFilter struct {
	numSamples       int         // number of samples in the particle filter
	percentResample  float64     // percent of particles that are resampled
	listParticles    []*Particle // list of our current particles
	xMin             float64     //TODO: Rework this to an initial estimate?
	xMax             float64
	yMin             float64 // TODO: Rework this to an initial estimate?
	yMax             float64
	EstimatedX       float64 // Where we think the x location is
	EstimatedY       float64 // Where we think the y location is
	EstimatedHeading float64 // Which way we think the rover is going
	maxWeight        float64 // maxWeight of the current iteration
	sumWeight        float64
	iteration        int           // current iteration of the filter
	locDistribution  distuv.Normal // error distribution for location
	angDistribution  distuv.Normal // error distribution for angle
	distDistribution distuv.Normal // error distribution for distance

}

// createpf creates an a particle filter object with various parameters
// numSamps: How many samples the particle filter has
// xMin: the minimum value the filter generates values for in the x direction
// xMax: the maximum value the filter generates values for in the x direction
// yMin: the minimum value the filter generates values for in the x direction
// yMax: the maximum value the filter generates values for in the x direction
// locError: sigma value for error in location
// distError: sigma value for error in distance
// angError: sigma value for error in angle
func CreatePF(numSamps int, percResamp, xMin, xMax, yMin, yMax, locError, distError, angError float64) *ParticleFilter {

	pf := &ParticleFilter{
		numSamples:      numSamps,
		percentResample: percResamp,
		listParticles:   []*Particle{},
		maxWeight:       0.0,
		iteration:       0, // how many times the filter has run

		locDistribution: distuv.Normal{
			Mu:    0,
			Sigma: locError,
		},
		angDistribution: distuv.Normal{
			Mu:    0,
			Sigma: angError,
		},
		distDistribution: distuv.Normal{
			Mu:    0,
			Sigma: distError,
		},
	}

	// creating initial random samples
	pf.createSampleList()

	return pf
}

// check if weight is greater than current max weight
func (pf *ParticleFilter) checkAndSetMaxWeight(weight float64, override bool) {
	if weight > pf.maxWeight {
		pf.maxWeight = weight
	}

	// when we want to reset to zero
	if override {
		pf.maxWeight = weight
	}
}

// createParticle creates a particle for use in a pf
func (pf *ParticleFilter) createParticle() Particle {
	xRange := pf.xMax - pf.xMax
	yRange := pf.yMax - pf.yMin
	x := pf.xMin + (xRange * rand.Float64())
	y := pf.yMin + (yRange * rand.Float64())
	heading := rand.Float64() * 2 * math.Pi
	p := Particle{X: x, Y: y, Weight: 0, Heading: heading}

	return p
}

// create initial sample list
func (pf *ParticleFilter) createSampleList() {
	for i := 0; i < pf.numSamples; i++ {
		p := pf.createParticle()
		pf.listParticles = append(pf.listParticles, &p)
	}
}

// CalculateWeights calculates the weight of each particle in the list by comparing the reading values
// to the location of the particles
func (pf *ParticleFilter) CalculateWeights(r Reading) {
	pf.checkAndSetMaxWeight(0.0, true)
	pf.sumWeight = 0
	for i := range pf.listParticles {
		p := pf.listParticles[i]

		// iterate over readings from each anchor
		// calculate length from each particle to anchor
		// assign weight to particle

		newWeight := r.CalculateWeight(*p)

		pf.checkAndSetMaxWeight(newWeight, false)
		p.UpdateWeight(newWeight)
		pf.sumWeight += newWeight
	}
}

// ResampleAndFuzz uses a resampling wheel to choose high weight particles
// fuzzes them to right around the chosen particles location
// and adds random particles in case we did not converge on the correct  answer
func (pf *ParticleFilter) ResampleAndFuzz() {
	pf.iteration += 1
	numResample := float64(pf.numSamples) * pf.percentResample
	numIntResample := int(numResample)
	numSpoof := pf.numSamples - numIntResample
	newParticleList := []*Particle{}

	for i := 0; i < pf.numSamples; i++ {
		p := pf.listParticles[i]
		p.UpdateWeight(p.Weight / pf.sumWeight)
	}

	// our new particle estimates
	xLoc := 0.0
	yLoc := 0.0
	heading := 0.0
	// Resample wheel code reused from OMSCS RAIT course Particle Filter Section
	// Originally developed by Sebastian Thrun
	for i := 0; i < numIntResample; i++ {
		beta := rand.Float64() * pf.maxWeight * 2
		startIdx := rand.Intn(numIntResample)
		for beta > 0 {
			p := pf.listParticles[startIdx]
			if p.Weight > beta {

				// creating normal dists around particle x and y
				// adding noise to that and reinserting into the list of particles

				newX := p.X + pf.locDistribution.Rand()

				newY := p.Y + pf.locDistribution.Rand()
				newHeading := p.Heading + pf.angDistribution.Rand()
				if newHeading < 0 {
					newHeading += (2.0 * math.Pi)
				} else if newHeading > 2.0*math.Pi {
					newHeading -= (2.0 * math.Pi)
				}

				newP := &Particle{
					X:       newX,
					Y:       newY,
					Heading: newHeading,
					Weight:  0,
				}

				xLoc += newX
				yLoc += newY
				heading += newHeading
				newParticleList = append(newParticleList, newP)
			}

			beta -= p.Weight
			startIdx = (startIdx + 1) % (pf.numSamples)
		}
	}

	// taking average of x and y locations and setting to estimated x and y locations
	xLoc /= float64(numIntResample)
	yLoc /= float64(numIntResample)
	heading /= float64(numIntResample)

	pf.EstimatedX = xLoc
	pf.EstimatedY = yLoc
	pf.EstimatedHeading = heading
	// adding some random samples incase we did not converge on correct answer
	for i := 0; i < numSpoof; i++ {
		p := pf.createParticle()
		newParticleList = append(newParticleList, &p)
	}

	pf.listParticles = newParticleList

}

func (pf *ParticleFilter) Move(dist, ang float64) {

	heading := 0.0
	x := 0.0
	y := 0.0

	for i := 0; i < pf.numSamples; i++ {
		particle := pf.listParticles[i]

		particle.Move(dist+pf.distDistribution.Rand(), ang+pf.angDistribution.Rand())

		heading += particle.Heading
		x += particle.X
		y += particle.Y

	}

	pf.EstimatedX = x / (float64(pf.numSamples) * pf.percentResample)
	pf.EstimatedY = y / (float64(pf.numSamples) * pf.percentResample)
	pf.EstimatedHeading = heading / (float64(pf.numSamples) * pf.percentResample)
}

// use pdf to get weight of particle's X or Y location
func CalculateNormDist(x, mu, sigma float64) float64 {
	xMuExponent := math.Pow(((x - mu) / sigma), 2.0)
	eulerExponent := math.Pow(math.E, (-.5 * xMuExponent))
	return ((1 / (sigma * math.Sqrt(2*math.Pi))) * eulerExponent)
}

func (pf *ParticleFilter) GetPosition() (float64, float64) {
	return pf.EstimatedX, pf.EstimatedY
}

func (pf *ParticleFilter) GetHeading() float64 {
	return pf.EstimatedHeading
}

func (pf *ParticleFilter) Update(r Reading) {

	pf.CalculateWeights(r)
	pf.ResampleAndFuzz()
}
