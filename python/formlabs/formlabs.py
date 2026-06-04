import asyncio
import struct

from bleak import BleakClient, BleakScanner


SVC = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2d"
CHR = "fa9b1d2c-3e4f-4a5b-9c6d-7e8f9a0b1c2e"
PKT = struct.Struct("<9f4b")


async def main():
    device = await BleakScanner.find_device_by_filter(
        lambda _, advert: SVC in (uuid.lower() for uuid in advert.service_uuids),
        timeout=20.0,
    )

    if device is None:
        raise SystemExit(f"no peripheral advertising {SVC}")

    async with BleakClient(device) as ble:
        await ble.start_notify(CHR, lambda _, data: print(*PKT.unpack(data)))
        await asyncio.Event().wait()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
