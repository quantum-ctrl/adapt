/**
 * CropBox - Interactive crop rectangle for data cropping
 * Supports dragging, resizing from 8 handles, and visual feedback
 */
export class CropBox {
    constructor(x, y, width, height, minSize = 20) {
        this.x = x;
        this.y = y;
        this.width = width;
        this.height = height;
        this.minSize = minSize;

        // Drag state
        this.isDragging = false;
        this.dragHandle = null; // 'move', 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'
        this.dragStartX = 0;
        this.dragStartY = 0;
        this.dragStartBoxX = 0;
        this.dragStartBoxY = 0;
        this.dragStartBoxWidth = 0;
        this.dragStartBoxHeight = 0;

        // Handle size in pixels
        this.handleSize = 8;
    }

    /**
     * Render the crop box with handles and mask overlay
     */
    render(ctx, canvasWidth, canvasHeight, margin) {
        // 1. Draw mask overlay (darken everything outside crop box)
        ctx.save();
        ctx.fillStyle = 'rgba(0, 0, 0, 0.45)';

        // Top region
        if (this.y > margin.top) {
            ctx.fillRect(margin.left, margin.top,
                canvasWidth - margin.left - margin.right,
                this.y - margin.top);
        }

        // Bottom region
        const cropBottom = this.y + this.height;
        const plotBottom = canvasHeight - margin.bottom;
        if (cropBottom < plotBottom) {
            ctx.fillRect(margin.left, cropBottom,
                canvasWidth - margin.left - margin.right,
                plotBottom - cropBottom);
        }

        // Left region
        if (this.x > margin.left) {
            ctx.fillRect(margin.left, this.y,
                this.x - margin.left,
                this.height);
        }

        // Right region
        const cropRight = this.x + this.width;
        const plotRight = canvasWidth - margin.right;
        if (cropRight < plotRight) {
            ctx.fillRect(cropRight, this.y,
                plotRight - cropRight,
                this.height);
        }

        ctx.restore();

        // 2. Draw crop box border
        ctx.save();
        ctx.strokeStyle = '#33aaff';
        ctx.lineWidth = 2;
        ctx.strokeRect(this.x, this.y, this.width, this.height);
        ctx.restore();

        // 3. Draw resize handles
        this.drawHandle(ctx, this.x, this.y); // NW
        this.drawHandle(ctx, this.x + this.width / 2, this.y); // N
        this.drawHandle(ctx, this.x + this.width, this.y); // NE
        this.drawHandle(ctx, this.x + this.width, this.y + this.height / 2); // E
        this.drawHandle(ctx, this.x + this.width, this.y + this.height); // SE
        this.drawHandle(ctx, this.x + this.width / 2, this.y + this.height); // S
        this.drawHandle(ctx, this.x, this.y + this.height); // SW
        this.drawHandle(ctx, this.x, this.y + this.height / 2); // W
    }

    /**
     * Draw a single resize handle
     */
    drawHandle(ctx, x, y) {
        ctx.save();
        ctx.fillStyle = '#33aaff';
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;

        const halfSize = this.handleSize / 2;
        ctx.fillRect(x - halfSize, y - halfSize, this.handleSize, this.handleSize);
        ctx.strokeRect(x - halfSize, y - halfSize, this.handleSize, this.handleSize);

        ctx.restore();
    }

    /**
     * Hit test to determine what part of the crop box was clicked
     * Returns: 'move', 'nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', or null
     */
    hitTest(mouseX, mouseY) {
        const tolerance = this.handleSize;

        // Check handles first (higher priority)
        // NW corner
        if (this.isNearPoint(mouseX, mouseY, this.x, this.y, tolerance)) {
            return 'nw';
        }
        // N edge
        if (this.isNearPoint(mouseX, mouseY, this.x + this.width / 2, this.y, tolerance)) {
            return 'n';
        }
        // NE corner
        if (this.isNearPoint(mouseX, mouseY, this.x + this.width, this.y, tolerance)) {
            return 'ne';
        }
        // E edge
        if (this.isNearPoint(mouseX, mouseY, this.x + this.width, this.y + this.height / 2, tolerance)) {
            return 'e';
        }
        // SE corner
        if (this.isNearPoint(mouseX, mouseY, this.x + this.width, this.y + this.height, tolerance)) {
            return 'se';
        }
        // S edge
        if (this.isNearPoint(mouseX, mouseY, this.x + this.width / 2, this.y + this.height, tolerance)) {
            return 's';
        }
        // SW corner
        if (this.isNearPoint(mouseX, mouseY, this.x, this.y + this.height, tolerance)) {
            return 'sw';
        }
        // W edge
        if (this.isNearPoint(mouseX, mouseY, this.x, this.y + this.height / 2, tolerance)) {
            return 'w';
        }

        // Check if inside box (for moving)
        if (mouseX >= this.x && mouseX <= this.x + this.width &&
            mouseY >= this.y && mouseY <= this.y + this.height) {
            return 'move';
        }

        return null;
    }

    /**
     * Check if point is near a target point
     */
    isNearPoint(x, y, targetX, targetY, tolerance) {
        return Math.abs(x - targetX) <= tolerance && Math.abs(y - targetY) <= tolerance;
    }

    /**
     * Get cursor style for a given handle
     */
    getCursor(handle) {
        const cursors = {
            'nw': 'nwse-resize',
            'n': 'ns-resize',
            'ne': 'nesw-resize',
            'e': 'ew-resize',
            'se': 'nwse-resize',
            's': 'ns-resize',
            'sw': 'nesw-resize',
            'w': 'ew-resize',
            'move': 'move'
        };
        return cursors[handle] || 'default';
    }

    /**
     * Start a drag operation
     */
    startDrag(handle, mouseX, mouseY) {
        this.isDragging = true;
        this.dragHandle = handle;
        this.dragStartX = mouseX;
        this.dragStartY = mouseY;
        this.dragStartBoxX = this.x;
        this.dragStartBoxY = this.y;
        this.dragStartBoxWidth = this.width;
        this.dragStartBoxHeight = this.height;
    }

    /**
     * Update drag operation
     */
    updateDrag(mouseX, mouseY) {
        if (!this.isDragging) return;

        const dx = mouseX - this.dragStartX;
        const dy = mouseY - this.dragStartY;

        if (this.dragHandle === 'move') {
            // Move the entire box
            this.x = this.dragStartBoxX + dx;
            this.y = this.dragStartBoxY + dy;
        } else {
            // Resize based on handle
            this.resizeFromHandle(this.dragHandle, dx, dy);
        }
    }

    /**
     * Resize box from a specific handle
     */
    resizeFromHandle(handle, dx, dy) {
        const origX = this.dragStartBoxX;
        const origY = this.dragStartBoxY;
        const origW = this.dragStartBoxWidth;
        const origH = this.dragStartBoxHeight;

        switch (handle) {
            case 'nw':
                this.x = origX + dx;
                this.y = origY + dy;
                this.width = origW - dx;
                this.height = origH - dy;
                break;
            case 'n':
                this.y = origY + dy;
                this.height = origH - dy;
                break;
            case 'ne':
                this.y = origY + dy;
                this.width = origW + dx;
                this.height = origH - dy;
                break;
            case 'e':
                this.width = origW + dx;
                break;
            case 'se':
                this.width = origW + dx;
                this.height = origH + dy;
                break;
            case 's':
                this.height = origH + dy;
                break;
            case 'sw':
                this.x = origX + dx;
                this.width = origW - dx;
                this.height = origH + dy;
                break;
            case 'w':
                this.x = origX + dx;
                this.width = origW - dx;
                break;
        }

        // Enforce minimum size
        if (this.width < this.minSize) {
            if (handle.includes('w')) {
                this.x = origX + origW - this.minSize;
            }
            this.width = this.minSize;
        }
        if (this.height < this.minSize) {
            if (handle.includes('n')) {
                this.y = origY + origH - this.minSize;
            }
            this.height = this.minSize;
        }
    }

    /**
     * End drag operation
     */
    endDrag() {
        this.isDragging = false;
        this.dragHandle = null;
    }

    /**
     * Constrain box to stay within bounds
     */
    constrain(minX, minY, maxX, maxY) {
        // Ensure box doesn't go outside bounds
        if (this.x < minX) this.x = minX;
        if (this.y < minY) this.y = minY;
        if (this.x + this.width > maxX) {
            if (this.width <= maxX - minX) {
                this.x = maxX - this.width;
            } else {
                this.x = minX;
                this.width = maxX - minX;
            }
        }
        if (this.y + this.height > maxY) {
            if (this.height <= maxY - minY) {
                this.y = maxY - this.height;
            } else {
                this.y = minY;
                this.height = maxY - minY;
            }
        }
    }

    /**
     * Get crop box bounds in plot coordinates (0-1 normalized)
     */
    getNormalizedBounds(margin, canvasWidth, canvasHeight) {
        const plotWidth = canvasWidth - margin.left - margin.right;
        const plotHeight = canvasHeight - margin.top - margin.bottom;

        const x0 = (this.x - margin.left) / plotWidth;
        const y0 = (this.y - margin.top) / plotHeight;
        const x1 = (this.x + this.width - margin.left) / plotWidth;
        const y1 = (this.y + this.height - margin.top) / plotHeight;

        return { x0, y0, x1, y1 };
    }
}
