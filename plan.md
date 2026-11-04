

## Proto Impulsive planner
- Determine relative change in elements (raan, aop, M)
- Pick maneuver time (of shortest or longest transfer)
  - Might be like several days
- Get rates required
- Get change in elements (p, e, i) needed
- calculate thrust for each

## ideas
- multiple shooting
- iLQR
- 



## Issues
- *Eclipse*
  - Issue
    - Planner averages thrust over time due to eclipse fraction
    - Propagator accounts for eclipse
    - If caught behind ellipse a lot, then adds eccentricity
  - Solution
    - After inclination added, based on eccentricity, add thrusting on other side or subtract thrusting on close side to circularize
  - Comments
    - Depends on mission requirements (eccentricity allowance)
- 

