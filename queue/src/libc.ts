import { dlopen, FFIType, ptr } from "bun:ffi";

const libc = dlopen("libc.so.6", {
  ioctl: { args: [FFIType.i32, FFIType.u64, FFIType.ptr], returns: FFIType.i32 },
  flock: { args: [FFIType.i32, FFIType.i32], returns: FFIType.i32 },
});

export const LOCK_EX = 2, LOCK_NB = 4, LOCK_UN = 8;
export const flock = (fd: number, op: number): boolean => libc.symbols.flock(fd, op) === 0;
export const ioctl = (fd: number, req: number, arg: Uint8Array | BigUint64Array): number => libc.symbols.ioctl(fd, req, ptr(arg));
export { ptr };
