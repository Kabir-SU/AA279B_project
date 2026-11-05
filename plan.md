

## Proto Impulsive planner
- Determine relative change in elements (raan, aop, M)
- Pick maneuver time (of shortest or longest transfer)
  - Might be like several days
- Get rates required
- Get change in elements (p, e, i) needed
- calculate thrust for each

## ideas
- multiple shooting
  - seeded by planner input
- iLQR
- 


## Steps
- Loose trajectory - initial pass 1
  - Duration, drift params, thrust-coast-thrust structure
- Redefine thrust distribution - initial pass 2
  - Correction based on how much it changes the ellipse
  - Trajectory-to-go heuristic based on orbital element error from target elements
- Optimization
  - 


## Ideas (post everything)
- *Strategy*
  - Continuous thrust transfer planner (with algorithms from Apa)
    <!-- - Apa, Ruggiero/Pergola, Varga -->
  - Discrete propagation for verification
- *Differences*
  - averaged dynamics and open loop control
- *Eclipse*
  - Issue
    - Planner averages thrust over time due to eclipse fraction
    - Propagator accounts for eclipse
    - If caught behind ellipse a lot, then adds eccentricity
  - Solution
    - After inclination added, based on eccentricity, add thrusting on other side or subtract thrusting on close side to circularize
  - Comments
    - Depends on mission requirements (eccentricity allowance)
    - Sumanth's and Apa
- 

