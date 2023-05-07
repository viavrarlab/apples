const cv = require("@techstark/opencv-js")
const ort = require("onnxruntime-web")


// for registering state proxy callbacks
const callbacks = {}
// internal state of the app
const state = new Proxy({}, {
    // intercept property set
    set: async (obj, key, value) => {
        // change property only if there's a difference
        if (obj[key] == value) return true
        // log and change property
        console.log(`state.${key} = ${obj[key]} => ${value}`)
        obj[key] = value
        // invoke callbacks and return
        try {
            if (key in callbacks) for (const callback of callbacks[key]) await callback(value)
        } catch (exc) {
            window.alert(`Not all combinations of options will work, some will fail, like this one - ${exc.toString()}`)
            throw exc
        }
        return true
    }
})


// generate simple load functions
function identity(value) {
    return value
}
function get_value(target) {
    return target.value
}
function get_checked(target) {
    return target.checked
}
function get_check_set_load(key, parse = identity, value = get_value, object = state) {
    function load(ev) {
        // simply check and set on object
        if (ev.target.reportValidity()) object[key] = parse(value(ev.target))
    }
    return load
}


// manual change dispatch
function change(el) {
    // simply dispatch with new event
    el.dispatchEvent(new Event("change"))
}


// plot histogram
function plot_hist(src_el, dst_el, rows = undefined, bins = 256, min = 0, max = 256) {
    // grab or read src
    const src = src_el instanceof cv.Mat ? src_el : cv.imread(src_el)
    // coalesce rows
    rows = (rows || undefined) ?? src.rows
    // need to pass src as vector
    const srcs = new cv.MatVector()
    srcs.push_back(src)
    // need to pass, not used
    const mask = new cv.Mat()
    // result histogram
    const hist = new cv.Mat()
    // result plot
    const dst = new cv.Mat.zeros(rows, bins, cv.CV_8UC3)
    try {
        // grab channels and calculate height of each histogram
        const channels = src.channels()
        const height = Math.trunc(rows / channels)
        // start with red and loop channels
        let color = new cv.Scalar(255, 0, 0)
        let channel = -1
        while (++channel < channels) {
            // calculate channel histogram and its bounds
            cv.calcHist(srcs, [channel], mask, hist, [bins], [min, max])
            const bounds = cv.minMaxLoc(hist)
            // calculate histogram plot floor height
            const floor = height * channel + height
            // loop bins
            let bin = -1
            while (++bin < bins) {
                // calculate plot value from hist
                // note that hist contains raw pixel counts, as such normalize with max and scale to height
                const value = Math.trunc(hist.data32F[bin] / bounds.maxVal * (height - 1))
                // don't draw points with no value
                if (!value) continue
                // draw a vertical line from bot to top
                const bot = new cv.Point(bin, floor)
                const top = new cv.Point(bin, floor - value)
                cv.line(dst, bot, top, color, 1)
            }
            // rotate color
            color = new cv.Scalar(color[2], color[0], color[1])
        }
        // show histogram
        cv.imshow(dst_el, dst)
    } finally {
        // deallocate all
        dst.delete()
        hist.delete()
        mask.delete()
        srcs.delete()
        if (!(src_el instanceof cv.Mat)) src.delete()
    }
}


// opencv.js has no extractChannel, define it
function extract_channel(src, dst, channel) {
    // allocate channels
    const srcs = new cv.MatVector()
    try {
        // split src and allocate temp
        cv.split(src, srcs)
        const temp = srcs.get(channel)
        try {
            // copy channel to dst
            temp.copyTo(dst)
        } finally {
            // deallocate temp
            temp.delete()
        }
    } finally {
        // deallocate channels
        srcs.delete()
    }
}


// opencv.js has no findNonZero, define it
function find_non_zero(src, dst) {
    // allocate temp
    const temp = new cv.Mat(src.rows * src.cols, 1, cv.CV_32FC2)
    try {
        // track non zero count and loop pixels
        let non_zero = 0
        let row = src.rows
        while (row--) {
            let col = src.cols
            while (col--) {
                // check pixel
                if (!src.data32F[src.cols * row + col]) continue
                // non zero, add to temp
                temp.data32F[non_zero++] = col
                temp.data32F[non_zero++] = row
            }
        }
        // copy if we found something
        if (non_zero) temp.rowRange(0, Math.floor(non_zero / 2)).copyTo(dst)
    } finally {
        // deallocate temp
        temp.delete()
    }
}


// opencv.js has no KeyPoint.convert, define it
function convert_points(points, dst) {
    // grab size, allocate dst and loop
    let point = points.size()
    dst.create(point, 1, cv.CV_32FC2)
    while (point--) {
        // grab point and copy coords
        let temp = points.get(point).pt
        dst.data32F[point * 2] = temp.x
        dst.data32F[point * 2 + 1] = temp.y
    }
}


// execute callback per channel
function exec_channels(src, dst, callback) {
    // grab and check channels
    const channels = src.channels()
    if (channels == 1) return callback(src, dst, 0)
    // got multiple channels, initialize vectors
    const srcs = new cv.MatVector()
    const dsts = new cv.MatVector()
    try {
        // split image and loop channels
        cv.split(src, srcs)
        let channel = -1
        while (++channel < channels) {
            // grab src and dst for channel
            const _src = srcs.get(channel)
            const _dst = _src.clone()
            try {
                // execute callback and save dst
                callback(_src, _dst, channel)
                dsts.push_back(_dst)
            } finally {
                // seems that vectors use protective copy, deallocate the ones we got
                _dst.delete()
                _src.delete()
            }
        }
        // merge channels back
        if (dst) cv.merge(dsts, dst)
    } finally {
        // deallocate vectors
        dsts.delete()
        srcs.delete()
    }
}


// execute image processing stage
async function process_img(dst_el, hist_el, check_hist, previous, next, process) {
    // check if we have previous stage
    if (!state[previous]) return
    // clone previous as next
    let mat_next = state[previous].clone()
    try {
        mat_next = await process(state[previous], mat_next)
        // show next and plot histogram
        cv.imshow(dst_el, mat_next)
        if (state[check_hist]) plot_hist(mat_next, hist_el)
    } catch (exc) {
        // failed, deallocate next
        mat_next.delete()
        throw exc
    }
    // deallocate previous and set next
    if (state[next]) state[next].delete()
    state[next] = mat_next
}


// generate histogram
function process_hist(hist_el, hist, mat) {
    // hide based on value and reload if histogram is required
    hist_el.hidden = !hist
    if (hist && mat) plot_hist(mat, hist_el)
}





// stage 1 - original input





// stream video from webcam
async function set_stream(stream, img_el, video_el, canvas_el, canvas_ctx, input_playback_rate_el, input_loop_el, load_video) {
    if (stream) {
        // we need stream, grab webcam
        video_el.srcObject = await navigator.mediaDevices.getUserMedia({ video: true, audio: false })
        // hide img el
        img_el.hidden = true
        // disable video specific controls, these make no sense for stream
        input_playback_rate_el.disabled = true
        input_loop_el.disabled = true
        // show video el
        video_el.hidden = false
        // load and play stream as video
        save_load(load_video)
        video_el.play()
    } else if (video_el.srcObject) {
        try {
            // need to disable stream, adjust canvas and copy the current frame
            canvas_el.width = video_el.clientWidth
            canvas_el.height = video_el.clientHeight
            canvas_ctx.drawImage(video_el, 0, 0, canvas_el.width, canvas_el.height)
            // load that frame as image, note that callbacks should do the rest here
            state.img_src = canvas_el.toDataURL()
        } finally {
            // potential fail, stop the stream anyway
            video_el.srcObject.getTracks().forEach(track => track.stop())
        }
    }
}


// load image and video
function load_src(src, type) {
    // src is already prepared, simply check type and set on internal state
    if (type.startsWith("image")) state.img_src = URL.createObjectURL(src)
    else if (type.startsWith("video")) state.video_src = URL.createObjectURL(src)
}
function set_img_src(img_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got img, pause and hide video el
    video_el.pause()
    video_el.hidden = true
    // disable video specific controls
    input_playback_rate_el.disabled = true
    input_loop_el.disabled = true
    // set and show img el
    img_el.src = img_src
    img_el.hidden = false
}
function set_video_src(video_src, img_el, video_el, input_playback_rate_el, input_loop_el) {
    // got video, hide img el
    img_el.hidden = true
    // enable video specific controls
    input_playback_rate_el.disabled = false
    input_loop_el.disabled = false
    // set and show video el
    video_el.src = video_src
    video_el.hidden = false
    // seems that we also need to reset video specific options
    video_el.playbackRate = state.playback_rate
    video_el.loop = state.loop
}


// load from file and url
function load_file(input_file_el, input_stream_el) {
    // check el and files and load the first
    if (input_file_el.reportValidity() && input_file_el.files.length) {
        // before load disable stream
        input_stream_el.checked = false
        change(input_stream_el)
        // then load
        load_src(input_file_el.files[0], input_file_el.files[0].type)
    }
}
async function load_url(input_url_el, input_stream_el, proxies) {
    // check el
    if (!input_url_el.reportValidity()) return
    // valid, grab and fetch url
    const url = input_url_el.value
    let response
    try {
        response = await fetch(url)
    } catch {
        // fetch fail, assume cors
        // shuffle proxies for load balance and loop
        proxies.sort(() => .5 - Math.random())
        let proxy_index = 0
        while (proxy_index < proxies.length) {
            try {
                // fetch from proxy
                response = await fetch(`${proxies[proxy_index++]}${url}`)
                // success, stop
                break
            } catch {
                // fail, next proxy
                continue
            }
        }
    }
    // check if we got response and load blob if so
    if (response) {
        // before load disable stream
        input_stream_el.checked = false
        change(input_stream_el)
        // then load
        load_src(await response.blob(), response.headers.get("Content-Type"))
    }
}


// load from drag & drop
function load_drop(ev, input_file_el, input_url_el, submit_fetch_el) {
    // stop browser from default and check if its files or urls
    ev.preventDefault()
    if (ev.dataTransfer.files.length) {
        // files, load through file el
        input_file_el.files = ev.dataTransfer.files
        // can't change value, need to fire change event manually
        change(input_file_el)
    } else if (ev.dataTransfer.items.length) ev.dataTransfer.items[0].getAsString(url => {
        // urls, load through url el
        input_url_el.value = url
        submit_fetch_el.click()
    })
}





// stage 2 - initial processing





// remember and load initial image
function save_load(load) {
    // store and execute load function
    state.load = load
    load()
}
function load_img(
    src_el, dst_el, canvas_el, canvas_ctx, width, height, hist_el, initial_hist_el,
    output_original_width_el, output_original_height_el, output_actual_fps_el) {
    // coalesce and check width and height
    canvas_el.width = state.width
    canvas_el.width ||= width
    canvas_el.height = state.height
    canvas_el.height ||= height
    if (!canvas_el.width || !canvas_el.height) return
    // resize and read new
    canvas_ctx.drawImage(src_el, 0, 0, canvas_el.width, canvas_el.height)
    let mat_initial = cv.imread(canvas_el)
    try {
        // plot original histogram and convert color space
        if (state.hist) plot_hist(mat_initial, hist_el, src_el.height)
        if (state.color_space) cv.cvtColor(mat_initial, mat_initial, cv[state.color_space])
        // extract channel
        if (state.channel) {
            // initialize vector
            const channels = new cv.MatVector()
            try {
                // split and grab channel
                cv.split(mat_initial, channels)
                const channel = channels.get(state.channel - 1)
                // deallocate old and swap
                mat_initial.delete()
                mat_initial = channel
            } finally {
                // deallocate vector
                channels.delete()
            }
        }
        // show new and plot histogram
        cv.imshow(dst_el, mat_initial)
        if (state.initial_hist) plot_hist(mat_initial, initial_hist_el)
    } catch (exc) {
        // failed, deallocate new
        mat_initial.delete()
        throw exc
    }
    // show original size
    output_original_width_el.innerHTML = src_el.videoWidth ?? src_el.width
    output_original_height_el.innerHTML = src_el.videoHeight ?? src_el.height
    output_actual_fps_el.innerHTML = 0
    // deallocate previous and set new
    if (state.mat_initial) state.mat_initial.delete()
    state.mat_initial = mat_initial
}


// process video at given fps
function load_play(video_el) {
    // simply set on internal state
    state.play = !video_el.paused && !video_el.ended
}
function set_play(play, output_actual_fps_el) {
    if (!play || state.playing) return
    // need to play and not playing, define and schedule callback
    function callback() {
        try {
            // mark start time
            const start = performance.now()
            // reload
            state.load()
            // calculate delta and show actual fps
            const delta = performance.now() - start
            output_actual_fps_el.innerHTML = state.fps ? Math.min(1000 / delta, state.fps) : 1000 / delta
            // calculate free time until next frame
            const free = state.fps > 0 ? 1000 / state.fps - delta : 0
            // schedule next callback if we're still playing
            state.playing = state.play ? setTimeout(callback, Math.max(0, free)) : undefined
        } catch (exc) {
            // exc, stop playing and rethrow
            state.playing = undefined
            throw exc
        }
    }
    state.playing = setTimeout(callback)
}





// stage 3 - blur





function blur_img(mat_previous, mat_next) {
    // blur based on type
    switch (state.blur) {
        case "blur":
            cv.blur(mat_previous, mat_next, new cv.Size(state.blur_kernel_width, state.blur_kernel_height))
            break
        case "GaussianBlur":
            cv.GaussianBlur(mat_previous, mat_next, new cv.Size(state.blur_kernel_width, state.blur_kernel_height), state.sigma_x, state.sigma_y)
            break
        case "medianBlur":
            cv.medianBlur(mat_previous, mat_next, state.blur_kernel_size)
            break
        case "bilateralFilter":
            exec_channels(mat_previous, mat_next, (src, dst) => cv.bilateralFilter(src, dst, state.blur_diameter, state.sigma_color, state.sigma_space))
            break
    }
    return mat_next
}


function set_blur(
    blur,
    input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
    input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el) {
    input_blur_kernel_width_el.disabled = true
    input_blur_kernel_height_el.disabled = true
    input_blur_kernel_size_el.disabled = true
    input_blur_diameter_el.disabled = true
    input_sigma_x_el.disabled = true
    input_sigma_y_el.disabled = true
    input_sigma_color_el.disabled = true
    input_sigma_space_el.disabled = true
    switch (blur) {
        case "blur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            break
        case "GaussianBlur":
            input_blur_kernel_width_el.disabled = false
            input_blur_kernel_height_el.disabled = false
            input_sigma_x_el.disabled = false
            input_sigma_y_el.disabled = false
            break
        case "medianBlur":
            input_blur_kernel_size_el.disabled = false
            break
        case "bilateralFilter":
            input_blur_diameter_el.disabled = false
            input_sigma_color_el.disabled = false
            input_sigma_space_el.disabled = false
            break
    }
}





// stage 4 - histogram equalization





function equalize_img(mat_previous, mat_next) {
    switch (state.equalization) {
        case "CLAHE":
            // allocate equalizer
            const clahe = new cv.CLAHE(state.equalization_clip, new cv.Size(state.equalization_columns, state.equalization_rows))
            try {
                // equalize each channel
                exec_channels(mat_previous, mat_next, (src, dst) => clahe.apply(src, dst))
            } finally {
                // deallocate equalizer
                clahe.delete()
            }
            break
        case "equalizeHist":
            // equalize each channel
            exec_channels(mat_previous, mat_next, (src, dst) => cv.equalizeHist(src, dst))
            break
    }
    return mat_next
}


function set_equalization(
    equalization,
    input_equalization_clip_el, input_equalization_rows_el, input_equalization_columns_el) {
    const disabled = equalization != "CLAHE"
    input_equalization_clip_el.disabled = disabled
    input_equalization_rows_el.disabled = disabled
    input_equalization_columns_el.disabled = disabled
}





// stage 5 - threshold





function threshold_img(mat_previous, mat_next) {
    if (state.threshold) {
        if (state.adaptive_threshold) {
            // adaptive
            exec_channels(mat_previous, mat_next, (src, dst) => cv.adaptiveThreshold(
                src, dst, state.threshold_max,
                cv[state.adaptive_threshold], cv[state.threshold],
                state.threshold_block, state.threshold_constant))
        } else if (state.optimal_threshold) {
            // optimal
            exec_channels(mat_previous, mat_next, (src, dst) => cv.threshold(
                src, dst, state.threshold_value, state.threshold_max,
                cv[state.threshold] | cv[state.optimal_threshold]))
        } else {
            // simple
            cv.threshold(
                mat_previous, mat_next, state.threshold_value, state.threshold_max,
                cv[state.threshold])
        }
    }
    return mat_next
}


function set_threshold(
    threshold,
    input_threshold_value_el, input_threshold_max_el,
    input_optimal_threshold_el, input_adaptive_threshold_el,
    input_threshold_block_el, input_threshold_constant_el) {
    input_threshold_value_el.disabled = true
    input_threshold_max_el.disabled = true
    input_optimal_threshold_el.disabled = true
    input_adaptive_threshold_el.disabled = true
    input_threshold_block_el.disabled = true
    input_threshold_constant_el.disabled = true
    switch (threshold) {
        case "THRESH_BINARY":
        case "THRESH_BINARY_INV":
            input_threshold_value_el.disabled = false
            input_threshold_max_el.disabled = false
            break
        case "THRESH_TRUNC":
        case "THRESH_TOZERO":
        case "THRESH_TOZERO_INV":
            input_threshold_value_el.disabled = false
            break
        default:
            return
    }
    if (state.optimal_threshold) {
        input_threshold_value_el.disabled = true
    } else {
        input_adaptive_threshold_el.disabled = false
    }
    if (state.adaptive_threshold) {
        input_threshold_value_el.disabled = true
        input_threshold_block_el.disabled = false
        input_threshold_constant_el.disabled = false
    } else {
        input_optimal_threshold_el.disabled = false
    }
}





// stage 6 - canny & contours





function canny_img(mat_previous, mat_next) {
    if (state.canny) exec_channels(
        mat_previous, mat_next,
        (src, dst) => cv.Canny(src, dst, state.canny_min, state.canny_max, state.canny_sobel, state.canny_l2))
    if (state.contours_mode) {
        // allocate zeros, note that cv.drawContours doesn't initialize
        const mat_contours = cv.Mat.zeros(mat_next.rows, mat_next.cols, cv.CV_8UC3)
        try {
            // process each channel, note that there is no dst due to cv.findContours requiring cv.CV_8UC1 src while cv.drawContours cv.CV_8UC3 dst
            exec_channels(mat_next, undefined, src => {
                // allocate contours and hierarchy, note that hierarchy is effectively unused
                const contours = new cv.MatVector()
                const hierarchy = new cv.Mat()
                try {
                    // find, loop and draw each contour with random color
                    cv.findContours(src, contours, hierarchy, cv.RETR_LIST, cv[state.contours_mode])
                    let contour = contours.size()
                    while (contour--) cv.drawContours(
                        mat_contours, contours, contour,
                        new cv.Scalar(Math.floor(Math.random() * 255), Math.floor(Math.random() * 255), Math.floor(Math.random() * 255)),
                        1, cv.LINE_8, hierarchy, 255)
                } finally {
                    // deallocate contours and hierarchy
                    hierarchy.delete()
                    contours.delete()
                }
            })
            // swap mat_contours with mat_next
            mat_next.delete()
            mat_next = mat_contours
        } catch (exc) {
            // failed, deallocate mat_contours
            mat_contours.delete()
            throw exc
        }
    }
    return mat_next
}


function set_canny(
    canny,
    input_canny_min_el, input_canny_max_el,
    input_canny_sobel_el, input_canny_l2_el) {
    const disabled = !canny
    input_canny_min_el.disabled = disabled
    input_canny_max_el.disabled = disabled
    input_canny_sobel_el.disabled = disabled
    input_canny_l2_el.disabled = disabled
}





// stage 7 - hough lines





function hough_img(mat_previous, mat_next) {
    if (state.hough) exec_channels(mat_previous, mat_next, (src, dst) => {
        // clear dst and initialize color since we draw manually
        dst.setTo(new cv.Scalar(0))
        const color = new cv.Scalar(255)
        // allocate lines and check hough
        const lines = new cv.Mat()
        try {
            if (state.hough == "HoughLines") {
                // standard, execute and loop lines
                cv.HoughLines(
                    src, lines, state.hough_rho, state.hough_theta * Math.PI / 180, state.hough_threshold,
                    state.hough_srn, state.hough_stn, state.hough_theta_min, state.hough_theta_max)
                let line = lines.rows * 2
                while ((line -= 2) >= 0) {
                    // draw line, see https://docs.opencv.org/4.7.0/d3/de6/tutorial_js_houghlines.html
                    const rho = lines.data32F[line]
                    const theta = lines.data32F[line + 1]
                    const a = Math.cos(theta)
                    const b = Math.sin(theta)
                    const x = a * rho
                    const y = b * rho
                    cv.line(
                        dst, new cv.Point(Math.floor(x - 1000 * b), Math.floor(y + 1000 * a)),
                        new cv.Point(Math.floor(x + 1000 * b), Math.floor(y - 1000 * a)), color)
                }
            } else {
                // probabilistic, execute, loop and draw lines
                cv.HoughLinesP(
                    src, lines, state.hough_rho, state.hough_theta * Math.PI / 180, state.hough_threshold,
                    state.hough_min_length, state.hough_max_gap)
                let line = lines.rows * 4
                while ((line -= 4) >= 0) cv.line(
                    dst, new cv.Point(lines.data32S[line], lines.data32S[line + 1]),
                    new cv.Point(lines.data32S[line + 2], lines.data32S[line + 3]), color)
            }
        } finally {
            // deallocate lines
            lines.delete()
        }
    })
    return mat_next
}


function set_hough(
    hough,
    input_hough_rho_el, input_hough_theta_el, input_hough_threshold_el,
    input_hough_srn_el, input_hough_stn_el,
    input_hough_theta_min_el, input_hough_theta_max_el,
    input_hough_min_length_el, input_hough_max_gap_el) {
    input_hough_rho_el.disabled = true
    input_hough_theta_el.disabled = true
    input_hough_threshold_el.disabled = true
    input_hough_srn_el.disabled = true
    input_hough_stn_el.disabled = true
    input_hough_theta_min_el.disabled = true
    input_hough_theta_max_el.disabled = true
    input_hough_min_length_el.disabled = true
    input_hough_max_gap_el.disabled = true
    switch (hough) {
        case "HoughLines":
            input_hough_rho_el.disabled = false
            input_hough_theta_el.disabled = false
            input_hough_threshold_el.disabled = false
            input_hough_srn_el.disabled = false
            input_hough_stn_el.disabled = false
            input_hough_theta_min_el.disabled = false
            input_hough_theta_max_el.disabled = false
            break
        case "HoughLinesP":
            input_hough_rho_el.disabled = false
            input_hough_theta_el.disabled = false
            input_hough_threshold_el.disabled = false
            input_hough_min_length_el.disabled = false
            input_hough_max_gap_el.disabled = false
            break
    }
}





// stage 8 - feature detection





function feat_img(mat_previous, mat_next) {
    // check feat and return
    if (!state.feat) return mat_next
    // check feat_points and initialize
    if (!state.feat_points) state.feat_points = []
    // grab channels and min from channels and feat_points
    const channels = mat_previous.channels()
    let channel = Math.min(channels, state.feat_points.length)
    // allocate missing, note that these are never deallocated
    while (channel++ < channels) state.feat_points.push(new cv.Mat())
    exec_channels(mat_previous, mat_next, (src, dst, channel) => {
        switch (state.feat) {
            case "cornerHarris":
                // harris corner detection, returns weight mask
                cv.cornerHarris(src, dst, state.feat_block_size, state.feat_sobel_size, state.feat_k)
                // threshold the mask based on quality, it's effectively invisible otherwise
                cv.threshold(dst, dst, state.feat_quality * cv.minMaxLoc(dst).maxVal, 255, cv.THRESH_BINARY)
                // convert corners to coords for flow
                find_non_zero(dst, state.feat_points[channel])
                break
            case "goodFeaturesToTrack":
                // shi-tomasi, grab corners and allocate mask
                const corners = state.feat_points[channel]
                const mask = new cv.Mat()
                try {
                    // find features
                    cv.goodFeaturesToTrack(
                        src, corners,
                        state.feat_max, state.feat_quality, state.feat_min_distance,
                        mask,
                        state.feat_block_size, state.feat_method == "cornerHarris", state.feat_k)
                    // draw features over dst
                    const color = new cv.Scalar(255)
                    let corner = corners.rows * 2
                    while ((corner -= 2) >= 0) cv.circle(dst, new cv.Point(corners.data32F[corner], corners.data32F[corner + 1]), 3, color)
                } finally {
                    // deallocate mask, note that we keep corners
                    mask.delete()
                }
                break
            case "FastFeatureDetector":
                // fast, allocate detector, key points and temp
                const fast = new cv.FastFeatureDetector()
                const fast_points = new cv.KeyPointVector()
                const fast_temp = new cv.Mat()
                try {
                    // configure and detect
                    fast.setThreshold(state.feat_threshold)
                    fast.setNonmaxSuppression(state.feat_suppression)
                    fast.detect(src, fast_points)
                    // draw features in temp, note that drawKeypoints results in rgb, whereas we need single channel, therefore extract it
                    cv.drawKeypoints(src, fast_points, fast_temp, new cv.Scalar(255, 0, 0))
                    extract_channel(fast_temp, dst, 0)
                    // convert key points to coords for flow
                    convert_points(fast_points, state.feat_points[channel])
                } finally {
                    // deallocate detector, key points and temp
                    fast_temp.delete()
                    fast_points.delete()
                    fast.delete()
                }
                break
            case "ORB":
                // orb, allocate detector, key points and temp
                const orb = new cv.ORB()
                const orb_points = new cv.KeyPointVector()
                const orb_temp = new cv.Mat()
                try {
                    // configure and detect
                    orb.setMaxFeatures(state.feat_max)
                    orb.setScaleFactor(state.feat_scale_factor)
                    orb.setNLevels(state.feat_levels)
                    orb.setEdgeThreshold(state.feat_edge_size)
                    orb.setFirstLevel(state.feat_first_level)
                    orb.setWTA_K(state.feat_wta_k)
                    orb.setPatchSize(state.feat_patch_size)
                    orb.setFastThreshold(state.feat_threshold)
                    orb.detect(src, orb_points)
                    // draw features in temp, note that drawKeypoints results in rgb, whereas we need single channel, therefore extract it
                    cv.drawKeypoints(src, orb_points, orb_temp, new cv.Scalar(255, 0, 0))
                    extract_channel(orb_temp, dst, 0)
                    // convert key points to coords for flow
                    convert_points(orb_points, state.feat_points[channel])
                } finally {
                    // deallocate detector, key points and temp
                    orb_temp.delete()
                    orb_points.delete()
                    orb.delete()
                }
                break
        }
    })
    return mat_next
}


function set_feat(
    feat,
    input_feat_block_size_el, input_feat_sobel_size_el,
    input_feat_k_el, input_feat_max_el, input_feat_quality_el,
    input_feat_min_distance_el, input_feat_method_el,
    input_feat_threshold_el, input_feat_suppression_el,
    input_feat_scale_factor_el, input_feat_levels_el,
    input_feat_edge_size_el, input_feat_first_level_el,
    input_feat_wta_k_el, input_feat_patch_size_el) {
    input_feat_block_size_el.disabled = true
    input_feat_sobel_size_el.disabled = true
    input_feat_k_el.disabled = true
    input_feat_max_el.disabled = true
    input_feat_quality_el.disabled = true
    input_feat_min_distance_el.disabled = true
    input_feat_method_el.disabled = true
    input_feat_threshold_el.disabled = true
    input_feat_suppression_el.disabled = true
    input_feat_scale_factor_el.disabled = true
    input_feat_levels_el.disabled = true
    input_feat_edge_size_el.disabled = true
    input_feat_first_level_el.disabled = true
    input_feat_wta_k_el.disabled = true
    input_feat_patch_size_el.disabled = true
    switch (feat) {
        case "cornerHarris":
            input_feat_block_size_el.disabled = false
            input_feat_sobel_size_el.disabled = false
            input_feat_k_el.disabled = false
            input_feat_quality_el.disabled = false
            break
        case "goodFeaturesToTrack":
            input_feat_block_size_el.disabled = false
            input_feat_k_el.disabled = false
            input_feat_max_el.disabled = false
            input_feat_quality_el.disabled = false
            input_feat_min_distance_el.disabled = false
            input_feat_method_el.disabled = false
            break
        case "FastFeatureDetector":
            input_feat_threshold_el.disabled = false
            input_feat_suppression_el.disabled = false
            break
        case "ORB":
            input_feat_max_el.disabled = false
            input_feat_scale_factor_el.disabled = false
            input_feat_levels_el.disabled = false
            input_feat_edge_size_el.disabled = false
            input_feat_first_level_el.disabled = false
            input_feat_wta_k_el.disabled = false
            input_feat_patch_size_el.disabled = false
            input_feat_threshold_el.disabled = false
            break
    }
}





// stage 9 - optical flow





function flow_img(mat_previous, mat_next) {
    // check flow and clear
    if (!state.flow) {
        if (state.mat_flow_previous) {
            // got previous, deallocate and clear
            state.mat_flow_previous.delete()
            state.mat_flow_previous = undefined
        }
        // deallocate masks and return
        while (state.flow_masks?.length) state.flow_masks.pop().delete()
        return mat_next
    }
    // check and initialize
    if (!state.mat_flow_previous) {
        // allocate and split channels
        state.mat_flow_previous = new cv.MatVector()
        cv.split(mat_previous, state.mat_flow_previous)
        // note that we return after initialization since we need at least two images
        return mat_next
    }
    // check and initialize
    if (!state.flow_masks) state.flow_masks = []
    // grab channels and min from channels and masks
    const channels = mat_previous.channels()
    let channel = Math.min(channels, state.flow_masks.length)
    // allocate missing masks
    while (channel++ < channels) state.flow_masks.push(new cv.Mat.zeros(mat_previous.rows, mat_previous.cols, cv.CV_8UC1))
    // check and exec farneback
    if (state.flow == "calcOpticalFlowFarneback") exec_channels(mat_previous, mat_next, (src, dst, channel) => {
        // grab corresponding channel from previous
        const src_previous = state.mat_flow_previous.get(channel)
        try {
            // calculate flow
            cv.calcOpticalFlowFarneback(
                src_previous, src,
                dst,
                state.flow_scale, state.flow_levels,
                state.flow_window_size, state.flow_iterations,
                state.flow_poly_n, state.flow_poly_sigma,
                state.flow_gaussian ? cv.OPTFLOW_FARNEBACK_GAUSSIAN : 0)
            // allocate channels and split
            const channels = new cv.MatVector()
            try {
                cv.split(dst, channels)
                // grab x and y
                const x = channels.get(0)
                const y = channels.get(1)
                // allocate angle and magnitude
                const angle = new cv.Mat()
                const magnitude = new cv.Mat()
                // allocate mat for multiplication, it seems that simple scalar doesn't work
                const mul = new cv.Mat(src.rows, src.cols, cv.CV_32FC1)
                mul.setTo(new cv.Scalar((1 / 360) * (180 / 255)))
                try {
                    // convert x and y to angle and magnitude, see https://docs.opencv.org/4.7.0/d4/dee/tutorial_optical_flow.html
                    cv.cartToPolar(x, y, magnitude, angle, true)
                    cv.normalize(magnitude, magnitude, 0, 1, cv.NORM_MINMAX)
                    cv.multiply(angle, mul, angle)
                    // we can display only a single channel
                    const flow = state.flow_layer == "magnitude" ? magnitude : angle
                    flow.copyTo(dst)
                } finally {
                    // deallocate all
                    mul.delete()
                    magnitude.delete()
                    angle.delete()
                    y.delete()
                    x.delete()
                }
            } finally {
                // deallocate channels
                channels.delete()
            }
        } finally {
            // deallocate defensive copy
            src_previous.delete()
        }
    })
    // no farneback, check and exec lucas-kanade, note that we also need feat_points
    else if (state.flow == "calcOpticalFlowPyrLK" && state.feat_points) exec_channels(mat_previous, mat_next, (src, dst, channel) => {
        // color for manual drawing
        const color = new cv.Scalar(255)
        // grab corresponding mask and points
        const mask = state.flow_masks[channel]
        const points_previous = state.feat_points[channel]
        // allocate new points for tracking
        const points = new cv.Mat()
        // grab corresponding channel from previous
        const src_previous = state.mat_flow_previous.get(channel)
        // allocate status and error, note that error is unused
        const status = new cv.Mat()
        const error = new cv.Mat()
        try {
            // calculate flow
            cv.calcOpticalFlowPyrLK(
                src_previous, src,
                points_previous, points,
                status, error,
                new cv.Size(state.flow_window_size, state.flow_window_size),
                state.flow_levels - 1)
            // count and loop points
            let ok = -1
            let point = -1
            while (++point < points.rows) {
                // check if we lost track
                if (!status.data[point]) {
                    // we did, mark the last known location
                    cv.circle(mask, new cv.Point(points_previous.data32F[point * 2], points_previous.data32F[point * 2 + 1]), 3, color)
                    continue
                }
                // still on track, draw trace
                cv.line(
                    mask, new cv.Point(points_previous.data32F[point * 2], points_previous.data32F[point * 2 + 1]),
                    new cv.Point(points.data32F[point * 2], points.data32F[point * 2 + 1]), color)
                // compress new points if we lost track at any point, note that this way we'll lose all our points eventually
                if (++ok != point) points.row(point).copyTo(points.row(ok))
            }
            // copy compressed points if we have track of any
            if (++ok) points.rowRange(0, ok).copyTo(points_previous)
            // draw traces over image
            cv.bitwise_or(dst, mask, dst)
        } finally {
            // deallocate all
            error.delete()
            status.delete()
            src_previous.delete()
            points.delete()
        }
    })
    // save for next frame
    cv.split(mat_previous, state.mat_flow_previous)
    return mat_next
}


function set_flow(
    flow,
    input_flow_scale_el, input_flow_levels_el,
    input_flow_window_size_el, input_flow_iterations_el,
    input_flow_poly_n_el, input_flow_poly_sigma_el,
    input_flow_gaussian_el, input_flow_layer_el) {
    input_flow_scale_el.disabled = true
    input_flow_levels_el.disabled = true
    input_flow_window_size_el.disabled = true
    input_flow_iterations_el.disabled = true
    input_flow_poly_n_el.disabled = true
    input_flow_poly_sigma_el.disabled = true
    input_flow_gaussian_el.disabled = true
    input_flow_layer_el.disabled = true
    switch (flow) {
        case "calcOpticalFlowFarneback":
            input_flow_scale_el.disabled = false
            input_flow_levels_el.disabled = false
            input_flow_window_size_el.disabled = false
            input_flow_iterations_el.disabled = false
            input_flow_poly_n_el.disabled = false
            input_flow_poly_sigma_el.disabled = false
            input_flow_gaussian_el.disabled = false
            input_flow_layer_el.disabled = false
            break
        case "calcOpticalFlowPyrLK":
            input_flow_levels_el.disabled = false
            input_flow_window_size_el.disabled = false
            break
    }
}





// stage 10 - classification





async function classification_img(mat_previous, mat_next) {
    if (!state.classification) return mat_next
    // need classification, initialize size and allocate channels
    const size = new cv.Size(320, 224)
    const channels = new cv.MatVector()
    try {
        // resize to match training size
        cv.resize(mat_previous, mat_next, size)
        // split and stack channels to match training shape
        cv.split(mat_next, channels)
        cv.vconcat(channels, mat_next)
        // load data to tensor and run model
        const input = new ort.Tensor("float32", Float32Array.from(mat_next.data), [3, size.height, size.width])
        const output = await state.classification_model.run({ "image": input })
        // reset mat_next
        mat_previous.copyTo(mat_next)
        // initialize scaling and color for manual drawing
        const factor_width = mat_next.cols / size.width
        const factor_height = mat_next.rows / size.height
        const color = new cv.Scalar(255, 0, 0)
        // grab classification data and loop
        const bboxes = output["bbox"].data
        const classes = output["class"].data
        const confidences = output["confidence"].data
        let index = -1
        while (++index < classes.length) {
            // check classification confidence
            if (confidences[index] < state.classification_threshold) continue
            // confident enough, scale and draw text and bbox
            const bbox = bboxes.slice(index * 4)
            cv.putText(
                mat_next, `${state.classification_labels[classes[index]]}: ${(confidences[index] * 100).toFixed(2)}%`,
                new cv.Point(bbox[0] * factor_width + 10, bbox[3] * factor_height - 10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv.rectangle(
                mat_next, new cv.Point(bbox[0] * factor_width, bbox[1] * factor_height),
                new cv.Point(bbox[2] * factor_width, bbox[3] * factor_height), color, 2)
        }
    } finally {
        // deallocate channels
        channels.delete()
    }
    return mat_next
}


async function set_classification(classification, input_classification_threshold_el) {
    input_classification_threshold_el.disabled = !classification
    if (!classification) return
    // got classification, load model
    state.classification_model = await ort.InferenceSession.create(classification)
    // and set respective labels
    switch (classification) {
        case "apples.onnx":
            state.classification_labels = ["fresh-apple", "damaged-apple"]
            break
        case "both.onnx":
            state.classification_labels = ["fresh-apple", "damaged-apple", "weed"]
            break
        case "weeds.onnx":
            state.classification_labels = ["weed"]
            break
    }
}





// initialization





// main entry point
function main() {
    //
    // stage 1 - original input
    //
    // grab IO UI elements
    const output_img_el = document.getElementById("output-img")
    const output_video_el = document.getElementById("output-video")
    const output_hist_el = document.getElementById("output-hist")
    const input_file_el = document.getElementById("input-file")
    const submit_fetch_el = document.getElementById("submit-fetch")
    const input_url_el = document.getElementById("input-url")
    const input_stream_el = document.getElementById("input-stream")
    const input_playback_rate_el = document.getElementById("input-playback-rate")
    const input_loop_el = document.getElementById("input-loop")
    const input_hist_el = document.getElementById("input-hist")
    const output_original_width_el = document.getElementById("output-original-width")
    const output_original_height_el = document.getElementById("output-original-height")
    const output_actual_fps_el = document.getElementById("output-actual-fps")
    // grab temp canvas and its context
    const temp_canvas_el = document.getElementById("temp-canvas")
    const temp_canvas_ctx = temp_canvas_el.getContext("2d")
    // proxies in case of cors problems
    const proxies = [
        // "https://cors-proxy.htmldriven.com/?url=",
        "https://corsproxy.io/?",
        // "https://crossorigin.me/",
        // "https://api.allorigins.win/raw?url=",
    ]
    const callback_load_img = () => load_img(
        output_img_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_img_el.width, output_img_el.height,
        output_hist_el, output_initial_hist_el,
        output_original_width_el, output_original_height_el, output_actual_fps_el)
    const callback_load_video = () => load_img(
        output_video_el, output_initial_el,
        temp_canvas_el, temp_canvas_ctx,
        output_video_el.clientWidth, output_video_el.clientHeight,
        output_hist_el, output_initial_hist_el,
        output_original_width_el, output_original_height_el, output_actual_fps_el)
    // register callbacks to react to internal state changes
    callbacks.play = [play => set_play(play, output_actual_fps_el)]
    callbacks.stream = [stream => set_stream(
        stream,
        output_img_el, output_video_el,
        temp_canvas_el, temp_canvas_ctx,
        input_playback_rate_el, input_loop_el,
        callback_load_video)]
    callbacks.playback_rate = [playback_rate => output_video_el.playbackRate = playback_rate]
    callbacks.loop = [loop => output_video_el.loop = loop]
    callbacks.hist = [hist => process_hist(output_hist_el, hist, temp_canvas_el)]
    // react to img/video src change fired by multiple methods
    callbacks.img_src = [img_src => set_img_src(img_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    callbacks.video_src = [video_src => set_video_src(video_src, output_img_el, output_video_el, input_playback_rate_el, input_loop_el)]
    // register callbacks to change internal state on element changes
    output_video_el.onplay = () => load_play(output_video_el)
    output_video_el.onpause = () => load_play(output_video_el)
    input_file_el.onchange = () => load_file(input_file_el, input_stream_el)
    submit_fetch_el.onclick = () => load_url(input_url_el, input_stream_el, proxies)
    input_stream_el.onchange = get_check_set_load("stream", identity, get_checked)
    input_playback_rate_el.onchange = get_check_set_load("playback_rate", parseFloat)
    input_loop_el.onchange = get_check_set_load("loop", identity, get_checked)
    input_hist_el.onchange = get_check_set_load("hist", identity, get_checked)
    // react to drag & drop on the entire document
    document.ondragover = ev => ev.preventDefault()
    document.ondrop = ev => load_drop(ev, input_file_el, input_url_el, submit_fetch_el)
    // react to the original src load
    // note that we remember which one we did last so that we can repeat it when stage config changes
    output_img_el.onload = () => save_load(callback_load_img)
    output_video_el.onloadeddata = () => save_load(callback_load_video)
    // also react to the user seeking to a new time in video
    output_video_el.onseeked = callback_load_video
    // fire element changes to initialize and sync internal state
    load_play(output_video_el)
    change(input_file_el)
    submit_fetch_el.click()
    change(input_stream_el)
    change(input_playback_rate_el)
    change(input_loop_el)
    change(input_hist_el)
    //
    // stage 2 - initial processing
    //
    const output_initial_el = document.getElementById("output-initial")
    const output_initial_hist_el = document.getElementById("output-initial-hist")
    const input_width_el = document.getElementById("input-width")
    const input_height_el = document.getElementById("input-height")
    const input_color_space_el = document.getElementById("input-color-space")
    const input_channel_el = document.getElementById("input-channel")
    const input_fps_el = document.getElementById("input-fps")
    const input_initial_hist_el = document.getElementById("input-initial-hist")
    const callback_load = () => state.load?.()
    callbacks.width = [callback_load]
    callbacks.height = [callback_load]
    callbacks.color_space = [callback_load]
    callbacks.channel = [callback_load]
    callbacks.initial_hist = [initial_hist => process_hist(output_initial_hist_el, initial_hist, state.mat_initial)]
    // enable fps only for video
    // note that this is monitered in set_play
    callbacks.img_src.push(() => input_fps_el.disabled = true)
    callbacks.video_src.push(() => input_fps_el.disabled = false)
    input_width_el.onchange = get_check_set_load("width", parseInt)
    input_height_el.onchange = get_check_set_load("height", parseInt)
    input_color_space_el.onchange = get_check_set_load("color_space")
    input_channel_el.onchange = get_check_set_load("channel", parseInt)
    input_fps_el.onchange = get_check_set_load("fps", parseInt)
    input_initial_hist_el.onchange = get_check_set_load("initial_hist", identity, get_checked)
    // dynamic options for color space based on whats available on cv
    Object.keys(cv).filter(key => key.startsWith("COLOR_")).forEach(key => {
        const input_color_space_option_el = document.createElement("option")
        input_color_space_option_el.value = key
        input_color_space_option_el.innerHTML = key.slice(6)
        input_color_space_el.appendChild(input_color_space_option_el)
    })
    // should be available, effectively get rid of alpha
    input_color_space_el.value = "COLOR_RGBA2RGB"
    change(input_width_el)
    change(input_height_el)
    change(input_color_space_el)
    change(input_channel_el)
    change(input_fps_el)
    change(input_initial_hist_el)
    //
    // stage 3 - blur
    //
    const output_blur_el = document.getElementById("output-blur")
    const output_blur_hist_el = document.getElementById("output-blur-hist")
    const input_blur_el = document.getElementById("input-blur")
    const input_blur_kernel_width_el = document.getElementById("input-blur-kernel-width")
    const input_blur_kernel_height_el = document.getElementById("input-blur-kernel-height")
    const input_blur_kernel_size_el = document.getElementById("input-blur-kernel-size")
    const input_blur_diameter_el = document.getElementById("input-blur-diameter")
    const input_sigma_x_el = document.getElementById("input-sigma-x")
    const input_sigma_y_el = document.getElementById("input-sigma-y")
    const input_sigma_color_el = document.getElementById("input-sigma-color")
    const input_sigma_space_el = document.getElementById("input-sigma-space")
    const input_blur_hist_el = document.getElementById("input-blur-hist")
    const callback_blur = () => process_img(
        output_blur_el, output_blur_hist_el,
        "blur_hist", "mat_initial", "mat_blur",
        blur_img)
    callbacks.mat_initial = [callback_blur]
    callbacks.blur = [blur => set_blur(
        blur,
        input_blur_kernel_width_el, input_blur_kernel_height_el, input_blur_kernel_size_el, input_blur_diameter_el,
        input_sigma_x_el, input_sigma_y_el, input_sigma_color_el, input_sigma_space_el), callback_blur]
    callbacks.blur_kernel_width = [callback_blur]
    callbacks.blur_kernel_height = [callback_blur]
    callbacks.blur_kernel_size = [callback_blur]
    callbacks.blur_diameter = [callback_blur]
    callbacks.sigma_x = [callback_blur]
    callbacks.sigma_y = [callback_blur]
    callbacks.sigma_color = [callback_blur]
    callbacks.sigma_space = [callback_blur]
    callbacks.blur_hist = [blur_hist => process_hist(output_blur_hist_el, blur_hist, state.mat_blur)]
    input_blur_el.onchange = get_check_set_load("blur")
    input_blur_kernel_width_el.onchange = get_check_set_load("blur_kernel_width", parseInt)
    input_blur_kernel_height_el.onchange = get_check_set_load("blur_kernel_height", parseInt)
    input_blur_kernel_size_el.onchange = get_check_set_load("blur_kernel_size", parseInt)
    input_blur_diameter_el.onchange = get_check_set_load("blur_diameter", parseInt)
    input_sigma_x_el.onchange = get_check_set_load("sigma_x", parseFloat)
    input_sigma_y_el.onchange = get_check_set_load("sigma_y", parseFloat)
    input_sigma_color_el.onchange = get_check_set_load("sigma_color", parseInt)
    input_sigma_space_el.onchange = get_check_set_load("sigma_space", parseInt)
    input_blur_hist_el.onchange = get_check_set_load("blur_hist", identity, get_checked)
    change(input_blur_el)
    change(input_blur_kernel_width_el)
    change(input_blur_kernel_height_el)
    change(input_blur_kernel_size_el)
    change(input_blur_diameter_el)
    change(input_sigma_x_el)
    change(input_sigma_y_el)
    change(input_sigma_color_el)
    change(input_sigma_space_el)
    change(input_blur_hist_el)
    //
    // stage 4 - histogram equalization
    //
    const output_equalization_el = document.getElementById("output-equalization")
    const output_equalization_hist_el = document.getElementById("output-equalization-hist")
    const input_equalization_el = document.getElementById("input-equalization")
    const input_equalization_clip_el = document.getElementById("input-equalization-clip")
    const input_equalization_rows_el = document.getElementById("input-equalization-rows")
    const input_equalization_columns_el = document.getElementById("input-equalization-columns")
    const input_equalization_hist_el = document.getElementById("input-equalization-hist")
    const callback_equalization = () => process_img(
        output_equalization_el, output_equalization_hist_el,
        "equalization_hist", "mat_blur", "mat_equalization",
        equalize_img)
    callbacks.mat_blur = [callback_equalization]
    callbacks.equalization = [equalization => set_equalization(
        equalization,
        input_equalization_clip_el, input_equalization_rows_el, input_equalization_columns_el), callback_equalization]
    callbacks.equalization_clip = [callback_equalization]
    callbacks.equalization_rows = [callback_equalization]
    callbacks.equalization_columns = [callback_equalization]
    callbacks.equalization_hist = [equalization_hist => process_hist(output_equalization_hist_el, equalization_hist, state.mat_equalization)]
    input_equalization_el.onchange = get_check_set_load("equalization")
    input_equalization_clip_el.onchange = get_check_set_load("equalization_clip", parseInt)
    input_equalization_rows_el.onchange = get_check_set_load("equalization_rows", parseInt)
    input_equalization_columns_el.onchange = get_check_set_load("equalization_columns", parseInt)
    input_equalization_hist_el.onchange = get_check_set_load("equalization_hist", identity, get_checked)
    change(input_equalization_el)
    change(input_equalization_clip_el)
    change(input_equalization_rows_el)
    change(input_equalization_columns_el)
    change(input_equalization_hist_el)
    //
    // stage 5 - threshold
    //
    const output_threshold_el = document.getElementById("output-threshold")
    const output_threshold_hist_el = document.getElementById("output-threshold-hist")
    const input_threshold_el = document.getElementById("input-threshold")
    const input_threshold_value_el = document.getElementById("input-threshold-value")
    const input_threshold_max_el = document.getElementById("input-threshold-max")
    const input_optimal_threshold_el = document.getElementById("input-optimal-threshold")
    const input_adaptive_threshold_el = document.getElementById("input-adaptive-threshold")
    const input_threshold_block_el = document.getElementById("input-threshold-block")
    const input_threshold_constant_el = document.getElementById("input-threshold-constant")
    const input_threshold_hist_el = document.getElementById("input-threshold-hist")
    const callback_threshold = () => process_img(
        output_threshold_el, output_threshold_hist_el,
        "threshold_hist", "mat_equalization", "mat_threshold",
        threshold_img)
    const callback_set_threshold = threshold => set_threshold(
        threshold,
        input_threshold_value_el, input_threshold_max_el,
        input_optimal_threshold_el, input_adaptive_threshold_el,
        input_threshold_block_el, input_threshold_constant_el)
    callbacks.mat_equalization = [callback_threshold]
    callbacks.threshold = [callback_set_threshold, callback_threshold]
    callbacks.threshold_value = [callback_threshold]
    callbacks.threshold_max = [callback_threshold]
    callbacks.optimal_threshold = [() => callback_set_threshold(state.threshold), callback_threshold]
    callbacks.adaptive_threshold = [() => callback_set_threshold(state.threshold), callback_threshold]
    callbacks.threshold_block = [callback_threshold]
    callbacks.threshold_constant = [callback_threshold]
    callbacks.threshold_hist = [threshold_hist => process_hist(output_threshold_hist_el, threshold_hist, state.mat_threshold)]
    input_threshold_el.onchange = get_check_set_load("threshold")
    input_threshold_value_el.onchange = get_check_set_load("threshold_value", parseInt)
    input_threshold_max_el.onchange = get_check_set_load("threshold_max", parseInt)
    input_optimal_threshold_el.onchange = get_check_set_load("optimal_threshold")
    input_adaptive_threshold_el.onchange = get_check_set_load("adaptive_threshold")
    input_threshold_block_el.onchange = get_check_set_load("threshold_block", parseInt)
    input_threshold_constant_el.onchange = get_check_set_load("threshold_constant", parseInt)
    input_threshold_hist_el.onchange = get_check_set_load("threshold_hist", identity, get_checked)
    change(input_threshold_el)
    change(input_threshold_value_el)
    change(input_threshold_max_el)
    change(input_optimal_threshold_el)
    change(input_adaptive_threshold_el)
    change(input_threshold_block_el)
    change(input_threshold_constant_el)
    change(input_threshold_hist_el)
    //
    // stage 6 - canny & contours
    //
    const output_canny_el = document.getElementById("output-canny")
    const output_canny_hist_el = document.getElementById("output-canny-hist")
    const input_canny_el = document.getElementById("input-canny")
    const input_canny_min_el = document.getElementById("input-canny-min")
    const input_canny_max_el = document.getElementById("input-canny-max")
    const input_canny_sobel_el = document.getElementById("input-canny-sobel")
    const input_canny_l2_el = document.getElementById("input-canny-l2")
    const input_contours_mode_el = document.getElementById("input-contours-mode")
    const input_canny_hist_el = document.getElementById("input-canny-hist")
    const callback_canny = () => process_img(
        output_canny_el, output_canny_hist_el,
        "canny_hist", "mat_threshold", "mat_canny",
        canny_img)
    callbacks.mat_threshold = [callback_canny]
    callbacks.canny = [canny => set_canny(
        canny,
        input_canny_min_el, input_canny_max_el,
        input_canny_sobel_el, input_canny_l2_el), callback_canny]
    callbacks.canny_min = [callback_canny]
    callbacks.canny_max = [callback_canny]
    callbacks.canny_sobel = [callback_canny]
    callbacks.canny_l2 = [callback_canny]
    callbacks.contours_mode = [callback_canny]
    callbacks.canny_hist = [canny_hist => process_hist(output_canny_hist_el, canny_hist, state.mat_canny)]
    input_canny_el.onchange = get_check_set_load("canny", identity, get_checked)
    input_canny_min_el.onchange = get_check_set_load("canny_min", parseInt)
    input_canny_max_el.onchange = get_check_set_load("canny_max", parseInt)
    input_canny_sobel_el.onchange = get_check_set_load("canny_sobel", parseInt)
    input_canny_l2_el.onchange = get_check_set_load("canny_l2", identity, get_checked)
    input_contours_mode_el.onchange = get_check_set_load("contours_mode")
    input_canny_hist_el.onchange = get_check_set_load("canny_hist", identity, get_checked)
    change(input_canny_el)
    change(input_canny_min_el)
    change(input_canny_max_el)
    change(input_canny_sobel_el)
    change(input_canny_l2_el)
    change(input_contours_mode_el)
    change(input_canny_hist_el)
    //
    // stage 7 - hough lines
    //
    const output_hough_el = document.getElementById("output-hough")
    const output_hough_hist_el = document.getElementById("output-hough-hist")
    const input_hough_el = document.getElementById("input-hough")
    const input_hough_rho_el = document.getElementById("input-hough-rho")
    const input_hough_theta_el = document.getElementById("input-hough-theta")
    const input_hough_threshold_el = document.getElementById("input-hough-threshold")
    const input_hough_srn_el = document.getElementById("input-hough-srn")
    const input_hough_stn_el = document.getElementById("input-hough-stn")
    const input_hough_theta_min_el = document.getElementById("input-hough-theta-min")
    const input_hough_theta_max_el = document.getElementById("input-hough-theta-max")
    const input_hough_min_length_el = document.getElementById("input-hough-min-length")
    const input_hough_max_gap_el = document.getElementById("input-hough-max-gap")
    const input_hough_hist_el = document.getElementById("input-hough-hist")
    const callback_hough = () => process_img(
        output_hough_el, output_hough_hist_el,
        "hough_hist", "mat_canny", "mat_hough",
        hough_img)
    callbacks.mat_canny = [callback_hough]
    callbacks.hough = [hough => set_hough(
        hough,
        input_hough_rho_el, input_hough_theta_el, input_hough_threshold_el,
        input_hough_srn_el, input_hough_stn_el,
        input_hough_theta_min_el, input_hough_theta_max_el,
        input_hough_min_length_el, input_hough_max_gap_el), callback_hough]
    callbacks.hough_rho = [callback_hough]
    callbacks.hough_theta = [callback_hough]
    callbacks.hough_threshold = [callback_hough]
    callbacks.hough_srn = [callback_hough]
    callbacks.hough_stn = [callback_hough]
    callbacks.hough_theta_min = [callback_hough]
    callbacks.hough_theta_max = [callback_hough]
    callbacks.hough_min_length = [callback_hough]
    callbacks.hough_max_gap = [callback_hough]
    callbacks.hough_hist = [hough_hist => process_hist(output_hough_hist_el, hough_hist, state.mat_hough)]
    input_hough_el.onchange = get_check_set_load("hough")
    input_hough_rho_el.onchange = get_check_set_load("hough_rho", parseInt)
    input_hough_theta_el.onchange = get_check_set_load("hough_theta", parseInt)
    input_hough_threshold_el.onchange = get_check_set_load("hough_threshold", parseInt)
    input_hough_srn_el.onchange = get_check_set_load("hough_srn", parseInt)
    input_hough_stn_el.onchange = get_check_set_load("hough_stn", parseInt)
    input_hough_theta_min_el.onchange = get_check_set_load("hough_theta_min", parseFloat)
    input_hough_theta_max_el.onchange = get_check_set_load("hough_theta_max", parseFloat)
    input_hough_min_length_el.onchange = get_check_set_load("hough_min_length", parseInt)
    input_hough_max_gap_el.onchange = get_check_set_load("hough_max_gap", parseInt)
    input_hough_hist_el.onchange = get_check_set_load("hough_hist", identity, get_checked)
    change(input_hough_el)
    change(input_hough_rho_el)
    change(input_hough_theta_el)
    change(input_hough_threshold_el)
    change(input_hough_srn_el)
    change(input_hough_stn_el)
    change(input_hough_theta_min_el)
    change(input_hough_theta_max_el)
    change(input_hough_min_length_el)
    change(input_hough_max_gap_el)
    change(input_hough_hist_el)
    //
    // stage 8 - feature detection
    //
    const output_feat_el = document.getElementById("output-feat")
    const output_feat_hist_el = document.getElementById("output-feat-hist")
    const input_feat_el = document.getElementById("input-feat")
    const input_feat_block_size_el = document.getElementById("input-feat-block-size")
    const input_feat_sobel_size_el = document.getElementById("input-feat-sobel-size")
    const input_feat_k_el = document.getElementById("input-feat-k")
    const input_feat_max_el = document.getElementById("input-feat-max")
    const input_feat_quality_el = document.getElementById("input-feat-quality")
    const input_feat_min_distance_el = document.getElementById("input-feat-min-distance")
    const input_feat_method_el = document.getElementById("input-feat-method")
    const input_feat_threshold_el = document.getElementById("input-feat-threshold")
    const input_feat_suppression_el = document.getElementById("input-feat-suppression")
    const input_feat_scale_factor_el = document.getElementById("input-feat-scale-factor")
    const input_feat_levels_el = document.getElementById("input-feat-levels")
    const input_feat_edge_size_el = document.getElementById("input-feat-edge-size")
    const input_feat_first_level_el = document.getElementById("input-feat-first-level")
    const input_feat_wta_k_el = document.getElementById("input-feat-wta-k")
    const input_feat_patch_size_el = document.getElementById("input-feat-patch-size")
    const input_feat_hist_el = document.getElementById("input-feat-hist")
    const callback_feat = () => process_img(
        output_feat_el, output_feat_hist_el,
        "feat_hist", "mat_hough", "mat_feat",
        feat_img)
    callbacks.mat_hough = [callback_feat]
    callbacks.feat = [feat => set_feat(
        feat,
        input_feat_block_size_el, input_feat_sobel_size_el,
        input_feat_k_el, input_feat_max_el, input_feat_quality_el,
        input_feat_min_distance_el, input_feat_method_el,
        input_feat_threshold_el, input_feat_suppression_el,
        input_feat_scale_factor_el, input_feat_levels_el,
        input_feat_edge_size_el, input_feat_first_level_el,
        input_feat_wta_k_el, input_feat_patch_size_el), callback_feat]
    callbacks.feat_block_size = [callback_feat]
    callbacks.feat_sobel_size = [callback_feat]
    callbacks.feat_k = [callback_feat]
    callbacks.feat_max = [callback_feat]
    callbacks.feat_quality = [callback_feat]
    callbacks.feat_min_distance = [callback_feat]
    callbacks.feat_method = [callback_feat]
    callbacks.feat_threshold = [callback_feat]
    callbacks.feat_suppression = [callback_feat]
    callbacks.feat_scale_factor = [callback_feat]
    callbacks.feat_levels = [callback_feat]
    callbacks.feat_edge_size = [callback_feat]
    callbacks.feat_first_level = [callback_feat]
    callbacks.feat_wta_k = [callback_feat]
    callbacks.feat_patch_size = [callback_feat]
    callbacks.feat_hist = [feat_hist => process_hist(output_feat_hist_el, feat_hist, state.mat_feat)]
    input_feat_el.onchange = get_check_set_load("feat")
    input_feat_block_size_el.onchange = get_check_set_load("feat_block_size", parseInt)
    input_feat_sobel_size_el.onchange = get_check_set_load("feat_sobel_size", parseInt)
    input_feat_k_el.onchange = get_check_set_load("feat_k", parseFloat)
    input_feat_max_el.onchange = get_check_set_load("feat_max", parseInt)
    input_feat_quality_el.onchange = get_check_set_load("feat_quality", parseFloat)
    input_feat_min_distance_el.onchange = get_check_set_load("feat_min_distance", parseInt)
    input_feat_method_el.onchange = get_check_set_load("feat_method")
    input_feat_threshold_el.onchange = get_check_set_load("feat_threshold", parseInt)
    input_feat_suppression_el.onchange = get_check_set_load("feat_suppression", identity, get_checked)
    input_feat_scale_factor_el.onchange = get_check_set_load("feat_scale_factor", parseFloat)
    input_feat_levels_el.onchange = get_check_set_load("feat_levels", parseInt)
    input_feat_edge_size_el.onchange = get_check_set_load("feat_edge_size", parseInt)
    input_feat_first_level_el.onchange = get_check_set_load("feat_first_level", parseInt)
    input_feat_wta_k_el.onchange = get_check_set_load("feat_wta_k", parseInt)
    input_feat_patch_size_el.onchange = get_check_set_load("feat_patch_size", parseInt)
    input_feat_hist_el.onchange = get_check_set_load("feat_hist", identity, get_checked)
    change(input_feat_el)
    change(input_feat_block_size_el)
    change(input_feat_sobel_size_el)
    change(input_feat_k_el)
    change(input_feat_max_el)
    change(input_feat_quality_el)
    change(input_feat_min_distance_el)
    change(input_feat_method_el)
    change(input_feat_threshold_el)
    change(input_feat_suppression_el)
    change(input_feat_scale_factor_el)
    change(input_feat_levels_el)
    change(input_feat_edge_size_el)
    change(input_feat_first_level_el)
    change(input_feat_wta_k_el)
    change(input_feat_patch_size_el)
    change(input_feat_hist_el)
    //
    // stage 9 - optical flow
    //
    const output_flow_el = document.getElementById("output-flow")
    const output_flow_hist_el = document.getElementById("output-flow-hist")
    const input_flow_el = document.getElementById("input-flow")
    const input_flow_scale_el = document.getElementById("input-flow-scale")
    const input_flow_levels_el = document.getElementById("input-flow-levels")
    const input_flow_window_size_el = document.getElementById("input-flow-window-size")
    const input_flow_iterations_el = document.getElementById("input-flow-iterations")
    const input_flow_poly_n_el = document.getElementById("input-flow-poly-n")
    const input_flow_poly_sigma_el = document.getElementById("input-flow-poly-sigma")
    const input_flow_gaussian_el = document.getElementById("input-flow-gaussian")
    const input_flow_layer_el = document.getElementById("input-flow-layer")
    const input_flow_hist_el = document.getElementById("input-flow-hist")
    const callback_flow = () => process_img(
        output_flow_el, output_flow_hist_el,
        "flow_hist", "mat_feat", "mat_flow",
        flow_img)
    callbacks.mat_feat = [callback_flow]
    callbacks.flow = [flow => set_flow(
        flow,
        input_flow_scale_el, input_flow_levels_el,
        input_flow_window_size_el, input_flow_iterations_el,
        input_flow_poly_n_el, input_flow_poly_sigma_el,
        input_flow_gaussian_el, input_flow_layer_el), callback_flow]
    callbacks.flow_scale = [callback_flow]
    callbacks.flow_levels = [callback_flow]
    callbacks.flow_window_size = [callback_flow]
    callbacks.flow_iterations = [callback_flow]
    callbacks.flow_poly_n = [callback_flow]
    callbacks.flow_poly_sigma = [callback_flow]
    callbacks.flow_gaussian = [callback_flow]
    callbacks.flow_layer = [callback_flow]
    callbacks.flow_hist = [flow_hist => process_hist(output_flow_hist_el, flow_hist, state.mat_flow)]
    input_flow_el.onchange = get_check_set_load("flow")
    input_flow_scale_el.onchange = get_check_set_load("flow_scale", parseFloat)
    input_flow_levels_el.onchange = get_check_set_load("flow_levels", parseInt)
    input_flow_window_size_el.onchange = get_check_set_load("flow_window_size", parseInt)
    input_flow_iterations_el.onchange = get_check_set_load("flow_iterations", parseInt)
    input_flow_poly_n_el.onchange = get_check_set_load("flow_poly_n", parseInt)
    input_flow_poly_sigma_el.onchange = get_check_set_load("flow_poly_sigma", parseFloat)
    input_flow_gaussian_el.onchange = get_check_set_load("flow_gaussian", identity, get_checked)
    input_flow_layer_el.onchange = get_check_set_load("flow_layer")
    input_flow_hist_el.onchange = get_check_set_load("flow_hist", identity, get_checked)
    change(input_flow_el)
    change(input_flow_scale_el)
    change(input_flow_levels_el)
    change(input_flow_window_size_el)
    change(input_flow_iterations_el)
    change(input_flow_poly_n_el)
    change(input_flow_poly_sigma_el)
    change(input_flow_gaussian_el)
    change(input_flow_layer_el)
    change(input_flow_hist_el)
    //
    // stage 10 - classification
    //
    const output_classification_el = document.getElementById("output-classification")
    const output_classification_hist_el = document.getElementById("output-classification-hist")
    const input_classification_el = document.getElementById("input-classification")
    const input_classification_threshold_el = document.getElementById("input-classification-threshold")
    const input_classification_hist_el = document.getElementById("input-classification-hist")
    const callback_classification = () => process_img(
        output_classification_el, output_classification_hist_el,
        "classification_hist", "mat_flow", "mat_classification",
        classification_img)
    callbacks.mat_flow = [callback_classification]
    callbacks.classification = [classification => set_classification(classification, input_classification_threshold_el), callback_classification]
    callbacks.classification_threshold = [callback_classification]
    callbacks.classification_hist = [classification_hist => process_hist(output_classification_hist_el, classification_hist, state.mat_classification)]
    input_classification_el.onchange = get_check_set_load("classification")
    input_classification_threshold_el.onchange = get_check_set_load("classification_threshold", parseFloat)
    input_classification_hist_el.onchange = get_check_set_load("classification_hist", identity, get_checked)
    change(input_classification_el)
    change(input_classification_threshold_el)
    change(input_classification_hist_el)
}


// seems that opencv does dynamic initialization, register main to be executed after that
cv.onRuntimeInitialized = main
