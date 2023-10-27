package particlefilter

import (
	"math"
	"math/rand"

	"gonum.org/v1/gonum/stat/distuv"
)

type Reading interface {
	CalculateWeight(Particle) float64
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

type ParticleFilter struct {
	NumSamples       int
	PercentResample  float64
	ListParticles    []*Particle
	Sigma            float64
	XLimit           float64
	YLimit           float64
	EstimatedX       float64
	EstimatedY       float64
	EstimatedHeading float64
	MaxWeight        float64
	iteration        int
	locDistribution  distuv.Normal
	angDistribution  distuv.Normal
	distDistribution distuv.Normal
}

// createpf creates an a particle filter object with various parameters
func CreatePF(numSamps int, percResamp, sigma, xLimit, yLimit, locError, distError, angError float64) *ParticleFilter {

	pf := &ParticleFilter{
		NumSamples:      numSamps,
		PercentResample: percResamp,
		ListParticles:   []*Particle{},
		Sigma:           sigma,
		XLimit:          xLimit,
		YLimit:          yLimit,
		MaxWeight:       0.0,
		iteration:       0,
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
	if weight > pf.MaxWeight {
		pf.MaxWeight = weight
	}

	// when we want to reset to zero
	if override {
		pf.MaxWeight = weight
	}
}

// createParticle creates a particle for use in a pf
func (pf *ParticleFilter) createParticle() Particle {
	x := rand.Float64() * pf.XLimit
	y := rand.Float64() * pf.YLimit
	heading := rand.Float64() * 2 * math.Pi
	p := Particle{X: x, Y: y, Weight: 0, Heading: heading}

	return p
}

// create initial sample list
func (pf *ParticleFilter) createSampleList() {
	for i := 0; i < pf.NumSamples; i++ {
		p := pf.createParticle()
		pf.ListParticles = append(pf.ListParticles, &p)
	}
}

// CalculateWeights calculates the weight of each particle in the list by comparing the reading values
// to the location of the particles
func (pf *ParticleFilter) CalculateWeights(r Reading) {
	pf.checkAndSetMaxWeight(0.0, true)
	for i := range pf.ListParticles {
		p := pf.ListParticles[i]

		// iterate over readings from each anchor
		// calculate length from each particle to anchor
		// assign weight to particle

		newWeight := r.CalculateWeight(*p)

		pf.checkAndSetMaxWeight(newWeight, false)
		p.UpdateWeight(newWeight)
	}
}

// ResampleAndFuzz uses a resampling wheel to choose high weight particles
// fuzzes them to right around the chosen particles location
// and adds random particles in case we did not converge on the correct  answer
func (pf *ParticleFilter) ResampleAndFuzz() {
	pf.iteration += 1
	numResample := float64(pf.NumSamples) * pf.PercentResample
	numIntResample := int(numResample)
	numSpoof := pf.NumSamples - numIntResample
	newParticleList := []*Particle{}

	// our new particle estimates
	xLoc := 0.0
	yLoc := 0.0

	// Resample wheel code reused from OMSCS RAIT course Particle Filter Section
	// Originally developed by Sebastian Thrun
	for i := 0; i < numIntResample; i++ {
		beta := rand.Float64() * pf.MaxWeight * 2
		startIdx := rand.Intn(numIntResample)
		for beta > 0 {
			p := pf.ListParticles[startIdx]
			if p.Weight > beta {

				// creating normal dists around particle x and y
				// adding noise to that and reinserting into the list of particles

				newX := p.X + pf.locDistribution.Rand()

				newY := p.Y + pf.locDistribution.Rand()

				newP := &Particle{
					X:      newX,
					Y:      newY,
					Weight: 0,
				}

				xLoc += newX
				yLoc += newY

				newParticleList = append(newParticleList, newP)
			}

			beta -= p.Weight
			startIdx = (startIdx + 1) % (pf.NumSamples)
		}
	}

	// taking average of x and y locations and setting to estimated x and y locations
	xLoc /= float64(numIntResample)
	yLoc /= float64(numIntResample)

	pf.EstimatedX = xLoc
	pf.EstimatedY = yLoc

	// adding some random samples incase we did not converge on correct answer
	for i := 0; i < numSpoof; i++ {
		p := pf.createParticle()
		newParticleList = append(newParticleList, &p)
	}

	pf.ListParticles = newParticleList

}

func (pf *ParticleFilter) Move(dist, ang float64) {

	heading := 0.0
	x := 0.0
	y := 0.0

	for i := range pf.ListParticles {
		particle := pf.ListParticles[i]

		dist += pf.distDistribution.Rand()
		ang += pf.angDistribution.Rand()

		particle.Heading += ang
		particle.X += dist * math.Cos(particle.Heading)
		particle.Y += dist * math.Sin(particle.Heading)

		heading += particle.Heading
		x += particle.X
		y += particle.Y
	}

	pf.EstimatedX = x / float64(len(pf.ListParticles))
	pf.EstimatedY = y / float64(len(pf.ListParticles))
	pf.EstimatedHeading = heading / float64(len(pf.ListParticles))

}

// use pdf to get weight of particle's X or Y location
func CalculateNormDist(x, mu, sigma float64) float64 {
	xMuExponent := math.Pow(((x - mu) / sigma), 2.0)
	eulerExponent := math.Pow(math.E, (-.5 * xMuExponent))
	return ((1 / (sigma * math.Sqrt(2*math.Pi))) * eulerExponent)
}
