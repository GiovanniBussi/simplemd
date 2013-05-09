!!!   This file is part of simplemd
!!!
!!!   simplemd is free software: you can redistribute it and/or modify
!!!   it under the terms of the GNU General Public License as published by
!!!   the Free Software Foundation, either version 3 of the License, or
!!!   (at your option) any later version.
!!!
!!!   Foobar is distributed in the hope that it will be useful,
!!!   but WITHOUT ANY WARRANTY; without even the implied warranty of
!!!   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!!!   GNU General Public License for more details.
!!!
!!!   You should have received a copy of the GNU General Public License
!!!   along with Foobar.  If not, see <http://www.gnu.org/licenses/>.

program simplemd
use routines
implicit none

integer              :: natoms          ! number of atoms
real,    allocatable :: positions(:,:)  ! atomic positions
real,    allocatable :: velocities(:,:) ! velocities
                                        ! was calculated last time
real,    allocatable :: masses(:)       ! masses
real,    allocatable :: forces(:,:)     ! forces   
real                 :: cell(3)         ! cell size

! neighbour list variables
! see Allen and Tildesey book for details
integer              :: listsize        ! size of the list array
integer, allocatable :: list(:)         ! neighbour list
integer, allocatable :: point(:)        ! pointer to neighbour list
real,    allocatable :: positions0(:,:) ! reference atomic positions, i.e. positions when the neighbour list

! input parameters
! all of them have a reasonable default value, set in read_input()
real           :: tstep          ! simulation timestep
real           :: temperature    ! temperature
real           :: friction       ! friction for Langevin dynamics (for NVE, use 0)
real           :: listcutoff     ! cutoff for neighbour list
real           :: forcecutoff    ! cutoff for forces
integer        :: nstep          ! number of steps
integer        :: nconfig        ! stride for output of configurations
integer        :: nstat          ! stride for output of statistics
integer        :: maxneighbour   ! maximum average number of neighbours per atom
integer        :: idum           ! seed
logical        :: wrapatoms      ! if true, atomic coordinates are written wrapped in minimal cell
character(256) :: inputfile      ! name of file with starting configuration (xyz)
character(256) :: outputfile     ! name of file with final configuration (xyz)
character(256) :: trajfile       ! name of the trajectory file (xyz)
character(256) :: statfile       ! name of the file with statistics


real    :: engkin         ! kinetic energy
real    :: engconf        ! configurational energy
real    :: engint         ! integral for conserved energy in Langevin dynamics

logical :: recompute_list ! control if the neighbour list have to be recomputed
integer :: istep          ! step counter
integer :: iatom

call read_input(temperature,tstep,friction,forcecutoff, &
                listcutoff,nstep,nconfig,nstat, &
                wrapatoms, &
                inputfile,outputfile,trajfile,statfile, &
                maxneighbour,idum)

! number of atoms is read from file inputfile
call read_natoms(inputfile,natoms)

! write the parameters in output so they can be checked
write(*,*) "Starting configuration           : ",trim(inputfile)
write(*,*) "Final configuration              : ",trim(outputfile)
write(*,*) "Number of atoms                  : ",natoms
write(*,*) "Temperature                      : ",temperature
write(*,*) "Time step                        : ",tstep
write(*,*) "Friction                         : ",friction
write(*,*) "Cutoff for forces                : ",forcecutoff
write(*,*) "Cutoff for neighbour list        : ",listcutoff
write(*,*) "Number of steps                  : ",nstep
write(*,*) "Stride for trajectory            : ",nconfig
write(*,*) "Trajectory file                  : ",trim(trajfile)
write(*,*) "Stride for statistics            : ",nstat
write(*,*) "Statistics file                  : ",trim(statfile)
write(*,*) "Max average number of neighbours : ",maxneighbour
write(*,*) "Seed                             : ",idum
write(*,*) "Are atoms wrapped on output?     : ",wrapatoms

! Since each atom pair is counted once, the total number of pairs
! will be half of the number of neighbours times the number of atoms
listsize=maxneighbour*natoms/2

! allocation of dynamical arrays
allocate(positions(3,natoms))
allocate(positions0(3,natoms))
allocate(velocities(3,natoms))
allocate(forces(3,natoms))
allocate(masses(natoms))
allocate(point(natoms))
allocate(list(listsize))

! masses are hard-coded to 1
masses=1.0

! energy integral initialized to 0
engint=0.0

! positions are read from file inputfile
call read_positions(inputfile,natoms,positions,cell)

! velocities are randomized according to temperature
call randomize_velocities(natoms,temperature,masses,velocities,idum)

! neighbour list are computed, and reference positions are saved
call compute_list(natoms,listsize,positions,cell,listcutoff,point,list)
write(*,*) "List size: ",point(natoms)-1
positions0=positions

! forces are computed before starting md
call compute_forces(natoms,listsize,positions,cell,forcecutoff,point,list,forces,engconf)

! here is the main md loop
! Langevin thermostat is applied before and after a velocity-Verlet integrator
! the overall structure is:
!   thermostat
!   update velocities
!   update positions
!   (eventually recompute neighbour list)
!   compute forces
!   update velocities
!   thermostat
!   (eventually dump output informations)
do istep=1,nstep
  call thermostat(natoms,masses,0.5*tstep,friction,temperature,velocities,engint,idum)

  do iatom=1,natoms
    velocities(:,iatom)=velocities(:,iatom)+forces(:,iatom)*0.5*tstep/masses(iatom)
  end do

  do iatom=1,natoms
    positions(:,iatom)=positions(:,iatom)+velocities(:,iatom)*tstep
  end do

! a check is performed to decide whether to recalculate the neighbour list
  call check_list(natoms,positions,positions0,listcutoff,forcecutoff,recompute_list)
  if(recompute_list) then
    call compute_list(natoms,listsize,positions,cell,listcutoff,point,list)
    positions0=positions
    write(*,*) "Neighbour list recomputed at step ",istep
    write(*,*) "List size: ",point(natoms)-1
  end if

  call compute_forces(natoms,listsize,positions,cell,forcecutoff,point,list,forces,engconf)

  do iatom=1,natoms
    velocities(:,iatom)=velocities(:,iatom)+forces(:,iatom)*0.5*tstep/masses(iatom)
  end do

  call thermostat(natoms,masses,0.5*tstep,friction,temperature,velocities,engint,idum)

! kinetic energy is calculated
  call compute_engkin(natoms,masses,velocities,engkin)

! eventually, write positions and statistics
  if(modulo(istep,nconfig)==0) call write_positions(trajfile,natoms,positions,cell,wrapatoms)
  if(modulo(istep,nstat)==0)   call write_statistics(statfile,istep,tstep,natoms,engkin,engconf,engint)

end do

call write_final_positions(outputfile,natoms,positions,cell,wrapatoms)
write(*,*) "Execution completed"

! deallocation of all allocatable array
deallocate(positions)
deallocate(velocities)
deallocate(forces)
deallocate(masses)
deallocate(positions0)
deallocate(point)
deallocate(list)

end program simplemd
