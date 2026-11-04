"""GPS deployment with full sunlight and no eclipse thrust switching.

Run: python -m scenarios.gps_deployment_no_eclipse --mode one
"""

from scenarios.gps_deployment import main


if __name__ == "__main__":
    main(default_eclipses=False)
