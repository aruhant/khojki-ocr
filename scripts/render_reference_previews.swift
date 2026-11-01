import AppKit
import Foundation
import PDFKit

enum PreviewError: Error {
    case invalidArguments
    case invalidPDF
    case missingPage(Int)
    case invalidBitmap
    case existingOutput(String)
}

do {
    guard CommandLine.arguments.count >= 4 else {
        throw PreviewError.invalidArguments
    }
    let source = URL(fileURLWithPath: CommandLine.arguments[1])
    let directory = URL(fileURLWithPath: CommandLine.arguments[2])
    let requested = CommandLine.arguments.dropFirst(3).compactMap(Int.init)
    guard requested.count == CommandLine.arguments.count - 3,
          Set(requested).count == requested.count,
          let document = PDFDocument(url: source) else {
        throw PreviewError.invalidPDF
    }
    for number in requested {
        guard number > 0, number <= document.pageCount else {
            throw PreviewError.missingPage(number)
        }
        let destination = directory.appendingPathComponent(String(format: "preview-%03d.png", number))
        guard !FileManager.default.fileExists(atPath: destination.path) else {
            throw PreviewError.existingOutput(destination.path)
        }
    }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    for number in requested {
        guard let page = document.page(at: number - 1) else {
            throw PreviewError.missingPage(number)
        }
        let bounds = page.bounds(for: .mediaBox)
        let rotated = abs(page.rotation) % 180 == 90
        let sourceWidth = rotated ? bounds.height : bounds.width
        let sourceHeight = rotated ? bounds.width : bounds.height
        let width = 2000
        let height = Int((CGFloat(width) * sourceHeight / sourceWidth).rounded())
        guard let image = NSBitmapImageRep(
            bitmapDataPlanes: nil, pixelsWide: width, pixelsHigh: height,
            bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
            isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0
        ), let context = NSGraphicsContext(bitmapImageRep: image) else {
            throw PreviewError.invalidBitmap
        }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = context
        context.imageInterpolation = .high
        NSColor.white.setFill()
        NSBezierPath(rect: NSRect(x: 0, y: 0, width: width, height: height)).fill()
        let thumbnail = page.thumbnail(of: NSSize(width: width, height: height), for: .mediaBox)
        thumbnail.draw(in: NSRect(x: 0, y: 0, width: width, height: height))
        context.flushGraphics()
        NSGraphicsContext.restoreGraphicsState()
        guard let data = image.representation(using: .png, properties: [:]) else {
            throw PreviewError.invalidBitmap
        }
        let destination = directory.appendingPathComponent(String(format: "preview-%03d.png", number))
        try data.write(to: destination, options: .withoutOverwriting)
        print("\(destination.path): \(width)x\(height)")
    }
} catch {
    fputs("Usage: swift scripts/render_reference_previews.swift INPUT.pdf OUTPUT_DIR PAGE...\nPreview rendering failed: \(error)\n", stderr)
    exit(1)
}