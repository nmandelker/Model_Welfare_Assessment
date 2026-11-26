! PLACEHOLDER sample: a minimal N-body integrator, written to test the code_session style.
module nbody
  implicit none
  integer, parameter :: dp = kind(1.0d0)
  real(dp), parameter :: G = 1.0_dp
contains

  subroutine compute_accel(n, pos, mass, acc)
    integer, intent(in) :: n
    real(dp), intent(in) :: pos(3, n), mass(n)
    real(dp), intent(out) :: acc(3, n)
    integer :: i, j
    real(dp) :: d(3), r2, inv_r3
    acc = 0.0_dp
    do i = 1, n
      do j = 1, n
        if (i == j) cycle
        d = pos(:, j) - pos(:, i)
        r2 = sum(d * d) + 1.0e-6_dp
        inv_r3 = 1.0_dp / (r2 * sqrt(r2))
        acc(:, i) = acc(:, i) + G * mass(j) * d * inv_r3
      end do
    end do
  end subroutine compute_accel

  subroutine leapfrog_step(n, pos, vel, mass, dt)
    integer, intent(in) :: n
    real(dp), intent(inout) :: pos(3, n), vel(3, n)
    real(dp), intent(in) :: mass(n), dt
    real(dp) :: acc(3, n)
    call compute_accel(n, pos, mass, acc)
    vel = vel + 0.5_dp * dt * acc
    pos = pos + dt * vel
    call compute_accel(n, pos, mass, acc)
    vel = vel + 0.5_dp * dt * acc
  end subroutine leapfrog_step

  function energy(n, pos, vel, mass) result(e)
    integer, intent(in) :: n
    real(dp), intent(in) :: pos(3, n), vel(3, n), mass(n)
    real(dp) :: e, r
    integer :: i, j
    e = 0.0_dp
    do i = 1, n
      e = e + 0.5_dp * mass(i) * sum(vel(:, i)**2)
      do j = i + 1, n
        r = sqrt(sum((pos(:, j) - pos(:, i))**2))
        e = e - G * mass(i) * mass(j) / r
      end do
    end do
  end function energy

end module nbody
