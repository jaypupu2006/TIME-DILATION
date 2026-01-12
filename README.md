## Twin Paradox Simulator (Special Relativity)

This project simulates the **Twin Paradox** using **Special Relativity** with natural units (c = 1).
The spacecraft follows a realistic velocity profile: linear acceleration to a maximum speed,
constant cruising, and linear deceleration, then mirrors the same process on the return trip.

At each Earth-frame time step, the traveler’s proper time is updated using:

dτ = √(1 − β²) dt

The simulation visualizes:
- Proper time τ versus Earth time t
- Velocity β = v/c versus time
- Position x(t) in the Earth frame

An animated dashboard displays real-time values of t, τ, γ, β, and x,
and prints the final time difference between Earth and the traveling twin when the simulation ends.

